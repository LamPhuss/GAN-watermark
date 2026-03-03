# ============================================================
# attacker.py
# Description: LLM Attacker with LoRA (G_θ) for the GAN
#   + Flash Attention 2 for faster inference/training
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for a linear layer.
    Original weight W is frozen: output = Wx + (BAx) * (alpha/r)
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_linear(x)

        # FIX: Cast LoRA weights to match input dtype (e.g. float16)
        # This is needed because model may be loaded in float16 but
        # LoRA params are initialized in float32
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)

        lora_out = self.lora_dropout(x) @ lora_A.T @ lora_B.T * self.scaling
        return original_out + lora_out


class CountStore:
    """Stores token transition statistics for learning watermark patterns."""

    def __init__(self, prevctx_width: int):
        self.prevctx_width = prevctx_width
        self.counts: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, ctx: tuple, tok: int, quantity: int = 1) -> None:
        self.counts[ctx][tok] += quantity

    def get(self, ctx: tuple) -> Dict[int, int]:
        return dict(self.counts.get(ctx, {}))

    def total_counts(self) -> int:
        return sum(sum(d.values()) for d in self.counts.values())


class StaticSpoofer:
    """
    Watermark Spoofer — multi-component scoring following paper Eq. 3-4.
    
    Combines:
      s*(T, [T1,T2,T3]) = w_abcd * s(T, {T1,T2,T3})     # full unordered
                         + w_partials * s(T, {T_min})      # partial (dominant token)
                         + w_empty * s(T, {})               # context-independent
    """

    def __init__(
        self,
        counts_base,       # CountStore
        counts_wm,         # CountStore
        prevctx_width: int,
        vocab_size: int,
        spoofer_strength: float = 7.5,   # ← Paper dùng 7.5, không phải 2.0!
        clip_at: float = 2.0,
        min_wm_count: int = 2,
        w_abcd: float = 2.0,
        w_partials: float = 1.0,
        w_empty: float = 0.5,
    ):
        self.counts_base = counts_base
        self.counts_wm = counts_wm
        self.prevctx_width = prevctx_width
        self.vocab_size = vocab_size
        self.spoofer_strength = spoofer_strength
        self.clip_at = clip_at
        self.min_wm_count = min_wm_count
        self.w_abcd = w_abcd
        self.w_partials = w_partials
        self.w_empty = w_empty
        
        # Cache for boosts
        self._cache = {}

    def get_boosts(self, ctx: tuple, device: str = "cpu") -> torch.Tensor:
        """
        Multi-component scoring matching paper's SpoofedProcessor.
        
        ctx: ordered tuple of prevctx_width token IDs, e.g. (100, 200, 300)
        Returns: boost vector [vocab_size] in [0, 1]
        """
        import torch
        
        # Check cache
        cache_key = (ctx, device)
        if cache_key in self._cache:
            return self._cache[cache_key]

        boosts = torch.zeros(self.vocab_size, device=device)
        total_w = 0.0

        # ── 1) Full context {A,B,C}→D — UNORDERED (sorted) ──
        ctx_sorted = tuple(sorted(ctx))
        score_full = self._score_for_ctx(ctx_sorted, ordered=False, device=device)
        if score_full is not None:
            boosts += self.w_abcd * score_full
            total_w += self.w_abcd

        # ── 2) Partial: find T_min and add {T_min}→D ──
        if self.w_partials > 0 and len(ctx) > 0:
            # Heuristic from paper Eq. 4: find token with strongest solo signal
            # In practice, just use all singles and let weighting handle it
            # Paper: find T_min via cosine similarity, simplified: try all
            best_partial = None
            best_sim = -1.0
            
            for tok in ctx:
                solo = self._score_for_ctx((tok,), ordered=False, device=device)
                if solo is not None:
                    # Use the one with highest norm (most informative)
                    norm = solo.norm().item()
                    if norm > best_sim:
                        best_sim = norm
                        best_partial = solo
            
            if best_partial is not None:
                boosts += self.w_partials * best_partial
                total_w += self.w_partials

        # ── 3) Empty context {}→D — context-independent ──
        if self.w_empty > 0:
            score_empty = self._score_for_ctx(tuple(), ordered=False, device=device)
            if score_empty is not None:
                boosts += self.w_empty * score_empty
                total_w += self.w_empty

        # ── Normalize ──
        if total_w > 0:
            boosts /= total_w

        self._cache[cache_key] = boosts
        return boosts

    def _score_for_ctx(self, ctx: tuple, ordered: bool, device: str):
        """
        Compute score vector for a single context.
        Score s(T, ctx) = clipped ratio of p_wm(T|ctx) / p_base(T|ctx).
        Returns None if insufficient data.
        """
        import torch

        try:
            c_base = self.counts_base.get(ctx, ordered=ordered)
            c_wm = self.counts_wm.get(ctx, ordered=ordered)
        except (ValueError, KeyError):
            return None

        if not c_wm:
            return None

        # Build tensors
        base_tensor = torch.zeros(self.vocab_size, device=device)
        wm_tensor = torch.zeros(self.vocab_size, device=device)
        for tok, cnt in c_base.items():
            if tok < self.vocab_size:
                base_tensor[tok] = cnt
        for tok, cnt in c_wm.items():
            if tok < self.vocab_size:
                wm_tensor[tok] = cnt

        # Probability masses
        mass_base = base_tensor / (base_tensor.sum() + 1e-6)
        mass_wm = wm_tensor / (wm_tensor.sum() + 1e-6)

        # Min data threshold (from paper)
        if len(ctx) == 0:
            # Empty context: use mass-based threshold
            min_thresh = max(1, round(0.00007 * base_tensor.sum().item()))
        else:
            min_thresh = self.min_wm_count

        enough_data = wm_tensor >= min_thresh
        base_nonzero = base_tensor > 0

        scores = torch.zeros_like(mass_wm)

        # Ratio for tokens with enough data in both
        valid = enough_data & base_nonzero
        scores[valid] = mass_wm[valid] / mass_base[valid]

        # Tokens only in WM (not in base) → very likely green
        only_wm = enough_data & ~base_nonzero
        max_ratio = scores[valid].max().item() if valid.any() else 1.0
        scores[only_wm] = max_ratio + 1e-3

        # Ratio < 1 → likely red → zero out
        scores[scores < 1.0] = 0.0

        # Clip to [0, clip_at] and normalize to [0, 1]
        scores = scores.clamp(0, self.clip_at)
        if scores.max() > 0:
            scores /= scores.max()

        return scores

def _load_model_with_flash_attn(model_name: str, device: str) -> AutoModelForCausalLM:
    """
    Load a HuggingFace causal LM with Flash Attention 2 if available.
    Falls back to default attention if flash_attn is not installed.

    Flash Attention 2 requirements:
      - flash-attn >= 2.0 installed
      - Model must be loaded in float16 or bfloat16 (not float32)
      - CUDA GPU required
    """
    load_kwargs = {
        "torch_dtype": torch.float16 if "cuda" in device else torch.float32,
    }

    # Try Flash Attention 2
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"[FlashAttn] flash_attn detected → using flash_attention_2")
    except ImportError:
        print(f"[FlashAttn] flash_attn not found → using default attention")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model


class AttackerLLM(nn.Module):
    """
    LLM Attacker with LoRA + Flash Attention 2.

    Wraps a pretrained LLM with LoRA adapters on attention layers.
    Only LoRA parameters are trained.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name

        print(f"[AttackerLLM] Loading {model_name}...")
        self.model = _load_model_with_flash_attn(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        self.lora_modules = {}
        self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

        self.model.to(device)
        print(f"[AttackerLLM] LoRA applied. Trainable params: {self.count_trainable_params():,}")

    def _apply_lora(self, r: int, alpha: int, dropout: float, target_modules: List[str]) -> None:
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, lora_layer)
                    self.lora_modules[name] = lora_layer

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_lora_params(self) -> List[nn.Parameter]:
        params = []
        for module in self.lora_modules.values():
            params.append(module.lora_A)
            params.append(module.lora_B)
        return params

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
        static_spoofer: Optional[StaticSpoofer] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        self.model.eval()

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        if static_spoofer is None:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            outputs = self._generate_with_spoofer(
                inputs, max_length, temperature, static_spoofer
            )

        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts, outputs

    def _generate_with_spoofer(
        self, inputs: dict, max_length: int,
        temperature: float, spoofer,
    ) -> torch.Tensor:
        """Generate text with static spoofer logit biasing."""
        import torch
        import torch.nn.functional as F
        
        input_ids = inputs['input_ids']
        batch_size = input_ids.size(0)

        for step in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]

            for b in range(batch_size):
                ctx_width = spoofer.prevctx_width
                if input_ids.size(1) >= ctx_width:
                    ctx = tuple(input_ids[b, -ctx_width:].cpu().tolist())
                    # get_boosts internally handles sorted + multi-component
                    boosts = spoofer.get_boosts(ctx, device=str(self.device))
                    logits[b] += spoofer.spoofer_strength * boosts[:logits.size(-1)]

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return input_ids

    def compute_log_probs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    def reinforce_loss(
        self,
        generated_ids: torch.LongTensor,
        rewards: torch.Tensor,
        prompt_lengths: torch.LongTensor,
        baseline: float = 0.5,
    ) -> torch.Tensor:
        log_probs = self.compute_log_probs(generated_ids)
        batch_size = generated_ids.size(0)
        seq_len = log_probs.size(1)

        mask = torch.zeros_like(log_probs)
        for b in range(batch_size):
            start = max(0, prompt_lengths[b].item() - 1)
            mask[b, start:] = 1.0

        masked_log_probs = (log_probs * mask).sum(dim=-1)
        advantage = rewards - baseline
        loss = -(advantage * masked_log_probs).mean()
        return loss


# ── Learning Module ──

class WatermarkLearner:
    """Learns watermark patterns from server responses."""

    def __init__(self, tokenizer, prevctx_width: int = 1):
        self.tokenizer = tokenizer
        self.prevctx_width = prevctx_width
        self.counts_wm = CountStore(prevctx_width)
        self.counts_base = CountStore(prevctx_width)

    def learn_from_watermarked(self, texts_wm: List[str]) -> None:
        toks_list = self.tokenizer(texts_wm)['input_ids']
        for toks in toks_list:
            for i in range(self.prevctx_width, len(toks)):
                ctx = tuple(toks[i - self.prevctx_width : i])
                self.counts_wm.add(ctx, toks[i], 1)

    def learn_from_baseline(self, texts_base: List[str]) -> None:
        toks_list = self.tokenizer(texts_base)['input_ids']
        for toks in toks_list:
            for i in range(self.prevctx_width, len(toks)):
                ctx = tuple(toks[i - self.prevctx_width : i])
                self.counts_base.add(ctx, toks[i], 1)

    def build_spoofer(self, vocab_size, spoofer_strength=7.5):
        return StaticSpoofer(
            counts_base=self.counts_base,
            counts_wm=self.counts_wm,
            prevctx_width=self.prevctx_width,
            vocab_size=vocab_size,
            spoofer_strength=spoofer_strength,
        )
