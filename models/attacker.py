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
    Static Spoofer based on learned count statistics.
    Computes boost scores indicating which tokens are "green" (watermarked).
    """

    def __init__(
        self,
        counts_base: CountStore,
        counts_wm: CountStore,
        prevctx_width: int,
        vocab_size: int,
        spoofer_strength: float = 2.0,
        clip_at: float = 2.0,
        min_wm_count: int = 2,
    ):
        self.counts_base = counts_base
        self.counts_wm = counts_wm
        self.prevctx_width = prevctx_width
        self.vocab_size = vocab_size
        self.spoofer_strength = spoofer_strength
        self.clip_at = clip_at
        self.min_wm_count = min_wm_count

    def get_boosts(self, ctx, device="cpu"):
        """
        Multi-component scoring following watermark-stealing paper (Eq. 3-4).

        Combines:
        - Full ordered context s(T, {A,B,C}) with weight w_full
        - Partial unordered contexts s(T, {A}), s(T, {B}), ... with weight w_partial
        - Empty context s(T, {}) with weight w_empty

        This is the key innovation that makes stealing work with large context widths.
        """
        import torch

        w_full = 2.0       # weight for full context (strongest signal)
        w_partial = 1.0    # weight for partial contexts
        w_empty = 0.5      # weight for empty/context-independent

        combined = torch.zeros(self.vocab_size, device=device)
        total_weight = 0.0

        # 1. Full ordered context: s(T, {A,B,C})
        score_full = self._compute_score_for_context(ctx, ordered=True, device=device)
        if score_full is not None:
            combined += w_full * score_full
            total_weight += w_full

        # 2. Partial contexts (unordered): s(T, {A}), s(T, {B}), s(T, {C})
        # Also try pairs: s(T, {A,B}), s(T, {A,C}), s(T, {B,C})
        if len(ctx) > 0:
            ctx_tokens = [t for t in ctx if t >= 0]  # filter wildcards
            # Singles
            for tok in ctx_tokens:
                partial_ctx = tuple(sorted([tok]))
                score_p = self._compute_score_for_context(partial_ctx, ordered=False, device=device)
                if score_p is not None:
                    combined += w_partial * score_p
                    total_weight += w_partial
            # Pairs (if prevctx_width >= 2)
            if len(ctx_tokens) >= 2:
                from itertools import combinations
                for pair in combinations(ctx_tokens, 2):
                    partial_ctx = tuple(sorted(pair))
                    score_p = self._compute_score_for_context(partial_ctx, ordered=False, device=device)
                    if score_p is not None:
                        combined += w_partial * 0.5 * score_p  # pairs get half weight
                        total_weight += w_partial * 0.5

        # 3. Empty context: s(T, {})
        score_empty = self._compute_score_for_context(tuple(), ordered=False, device=device)
        if score_empty is not None:
            combined += w_empty * score_empty
            total_weight += w_empty

        if total_weight > 0:
            combined /= total_weight

        return combined


def _compute_score_for_context(self, ctx, ordered, device):
    """Compute score vector for a single context. Returns None if no data."""
    import torch

    try:
        c_base = self.counts_base.get(ctx, ordered=ordered)
        c_wm = self.counts_wm.get(ctx, ordered=ordered)
    except (ValueError, KeyError):
        return None

    if not c_wm:
        return None

    base_tensor = torch.zeros(self.vocab_size, device=device)
    wm_tensor = torch.zeros(self.vocab_size, device=device)
    for tok, cnt in c_base.items():
        if tok < self.vocab_size:
            base_tensor[tok] = cnt
    for tok, cnt in c_wm.items():
        if tok < self.vocab_size:
            wm_tensor[tok] = cnt

    mass_base = base_tensor / (base_tensor.sum() + 1e-6)
    mass_wm = wm_tensor / (wm_tensor.sum() + 1e-6)

    # Score: ratio of WM mass to base mass, clipped to [0, clip_at]
    enough_data = wm_tensor >= self.min_wm_count
    base_nonzero = base_tensor > 0

    scores = torch.zeros_like(mass_wm)
    valid = enough_data & base_nonzero
    scores[valid] = mass_wm[valid] / mass_base[valid]

    # Tokens only in WM (not in base) → likely green
    only_wm = enough_data & ~base_nonzero
    if scores[valid].numel() > 0 and scores[valid].max() > 0:
        scores[only_wm] = scores[valid].max() + 1e-3
    else:
        scores[only_wm] = 1.0

    # Threshold: ratio < 1 means token appears LESS in WM → likely red
    scores[scores < 1.0] = 0.0

    # Clip and normalize to [0, 1]
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
        temperature: float, spoofer: StaticSpoofer,
    ) -> torch.Tensor:
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

    def build_spoofer(
        self, vocab_size: int, spoofer_strength: float = 2.0,
    ) -> StaticSpoofer:
        return StaticSpoofer(
            counts_base=self.counts_base,
            counts_wm=self.counts_wm,
            prevctx_width=self.prevctx_width,
            vocab_size=vocab_size,
            spoofer_strength=spoofer_strength,
        )
