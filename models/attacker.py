# ============================================================
# attacker.py
# Description: LLM Attacker with LoRA (G_θ) for the GAN
#   Learns to generate spoofed watermark text that fools D
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
    
    Only the low-rank matrices A and B are trainable.
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

        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        original_out = self.original_linear(x)
        # LoRA delta
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original_out + lora_out


class CountStore:
    """
    Simplified CountStore adapted from watermark-stealing.
    Stores token transition statistics for learning watermark patterns.
    """

    def __init__(self, prevctx_width: int):
        self.prevctx_width = prevctx_width
        self.counts: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, ctx: tuple, tok: int, quantity: int = 1) -> None:
        """Add a count for context->token."""
        self.counts[ctx][tok] += quantity

    def get(self, ctx: tuple) -> Dict[int, int]:
        """Get all counts for a given context."""
        return dict(self.counts.get(ctx, {}))

    def total_counts(self) -> int:
        return sum(sum(d.values()) for d in self.counts.values())


class StaticSpoofer:
    """
    Static Spoofer based on learned count statistics (from watermark-stealing's SpoofedProcessor).
    
    Given context tokens, computes boost scores that indicate which tokens
    are more likely to be "green" (watermarked) vs "red" (non-watermarked).
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

    def get_boosts(self, ctx: tuple, device: str = "cpu") -> torch.Tensor:
        """
        Compute boost vector for the given context.
        
        Returns:
            (vocab_size,) tensor of boost values in [0, 1]
        """
        c_base = self.counts_base.get(ctx)
        c_wm = self.counts_wm.get(ctx)

        # Build tensors
        base_tensor = torch.zeros(self.vocab_size, device=device)
        wm_tensor = torch.zeros(self.vocab_size, device=device)
        for tok, cnt in c_base.items():
            if tok < self.vocab_size:
                base_tensor[tok] = cnt
        for tok, cnt in c_wm.items():
            if tok < self.vocab_size:
                wm_tensor[tok] = cnt

        # Mass
        mass_base = base_tensor / (base_tensor.sum() + 1e-6)
        mass_wm = wm_tensor / (wm_tensor.sum() + 1e-6)

        # Ratios
        enough_data_mask = wm_tensor >= self.min_wm_count
        base_zero_mask = base_tensor == 0

        ratios = torch.zeros_like(mass_wm)
        core_mask = enough_data_mask & ~base_zero_mask
        ratios[core_mask] = mass_wm[core_mask] / mass_base[core_mask]
        ratios[enough_data_mask & base_zero_mask] = max(1, ratios.max().item()) + 1e-3

        # Normalize
        ratios[ratios < 1] = 0
        ratios[ratios > self.clip_at] = self.clip_at
        ratios /= self.clip_at + 1e-8

        if ratios.max() > 0:
            ratios = ratios / (ratios.max() + 1e-8)

        return ratios


class AttackerLLM(nn.Module):
    """
    LLM Attacker with LoRA.
    
    Wraps a pretrained LLM (e.g., OPT-1.3B or Gemma-2B) with LoRA adapters
    on the attention layers. Only LoRA parameters are trained.
    
    During adversarial training, uses REINFORCE policy gradient:
        ∇θ J(θ) ≈ E[R_total · ∇θ log G_θ(y_t | y_{<t})]
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

        # Load pretrained LLM
        print(f"[AttackerLLM] Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA to target modules
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]
        
        self.lora_modules = {}
        self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

        self.model.to(device)
        print(f"[AttackerLLM] LoRA applied. Trainable params: {self.count_trainable_params():,}")

    def _apply_lora(self, r: int, alpha: int, dropout: float, target_modules: List[str]) -> None:
        """Apply LoRA adapters to target attention modules."""
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                    # Replace the module in the parent
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, lora_layer)
                    self.lora_modules[name] = lora_layer

    def count_trainable_params(self) -> int:
        """Count number of trainable parameters (LoRA only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_lora_params(self) -> List[nn.Parameter]:
        """Get only the LoRA parameters for the optimizer."""
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
        """
        Generate text from prompts.
        
        If static_spoofer is provided, boost logits for green tokens during generation.
        
        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy
            static_spoofer: Optional spoofer for boosted generation
        
        Returns:
            Tuple of (generated texts, token IDs tensor)
        """
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
            # Manual generation with spoofer boost
            outputs = self._generate_with_spoofer(
                inputs, max_length, temperature, static_spoofer
            )
        
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts, outputs

    def _generate_with_spoofer(
        self,
        inputs: dict,
        max_length: int,
        temperature: float,
        spoofer: StaticSpoofer,
    ) -> torch.Tensor:
        """Generate token-by-token with spoofer boost applied."""
        input_ids = inputs['input_ids']
        batch_size = input_ids.size(0)
        
        for step in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # (batch, vocab)
            
            # Apply spoofer boost per batch
            for b in range(batch_size):
                ctx_width = spoofer.prevctx_width
                if input_ids.size(1) >= ctx_width:
                    ctx = tuple(input_ids[b, -ctx_width:].cpu().tolist())
                    boosts = spoofer.get_boosts(ctx, device=str(self.device))
                    logits[b] += spoofer.spoofer_strength * boosts[:logits.size(-1)]
            
            # Sample
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if all sequences have EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        return input_ids

    def compute_log_probs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log P(y_t | y_{<t}) for each token in the sequence.
        Used for both PPL computation and REINFORCE gradient.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
        
        Returns:
            (batch, seq_len-1) log probabilities of each token given its prefix
        """
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab)
        
        # Gather log probs of actual next tokens
        target_ids = input_ids[:, 1:]  # (batch, seq_len-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather the log prob of each actual token
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len-1)
        
        return token_log_probs

    def reinforce_loss(
        self,
        generated_ids: torch.LongTensor,
        rewards: torch.Tensor,
        prompt_lengths: torch.LongTensor,
        baseline: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute REINFORCE (policy gradient) loss.
        
        Loss = -E[(R - baseline) * sum(log P(y_t | y_{<t}))]
        
        Only compute gradients for tokens AFTER the prompt (generated tokens).
        
        Args:
            generated_ids: (batch, seq_len) full sequences (prompt + generated)
            rewards: (batch,) reward for each sequence
            prompt_lengths: (batch,) length of each prompt
            baseline: Scalar baseline for variance reduction
        
        Returns:
            Scalar loss for gradient ascent (negate for gradient descent)
        """
        # Compute log probs for the whole sequence
        log_probs = self.compute_log_probs(generated_ids)  # (batch, seq_len-1)
        
        batch_size = generated_ids.size(0)
        seq_len = log_probs.size(1)
        
        # Create mask: only count generated tokens (after prompt)
        mask = torch.zeros_like(log_probs)
        for b in range(batch_size):
            start = max(0, prompt_lengths[b].item() - 1)  # -1 because log_probs is shifted
            mask[b, start:] = 1.0
        
        # Masked sum of log probs per sequence
        masked_log_probs = (log_probs * mask).sum(dim=-1)  # (batch,)
        
        # REINFORCE: Loss = -E[(R - b) * log_prob]
        advantage = rewards - baseline
        loss = -(advantage * masked_log_probs).mean()
        
        return loss


# ── Learning Module (from watermark-stealing) ──

class WatermarkLearner:
    """
    Learns watermark patterns from server responses.
    Adapted from OurAttacker._learn_fast() in watermark-stealing.
    """

    def __init__(self, tokenizer, prevctx_width: int = 1):
        self.tokenizer = tokenizer
        self.prevctx_width = prevctx_width
        self.counts_wm = CountStore(prevctx_width)
        self.counts_base = CountStore(prevctx_width)

    def learn_from_watermarked(self, texts_wm: List[str]) -> None:
        """Learn token patterns from watermarked texts (fast mode)."""
        toks_list = self.tokenizer(texts_wm)['input_ids']
        for toks in toks_list:
            for i in range(self.prevctx_width, len(toks)):
                ctx = tuple(toks[i - self.prevctx_width : i])
                self.counts_wm.add(ctx, toks[i], 1)

    def learn_from_baseline(self, texts_base: List[str]) -> None:
        """Learn token patterns from non-watermarked texts."""
        toks_list = self.tokenizer(texts_base)['input_ids']
        for toks in toks_list:
            for i in range(self.prevctx_width, len(toks)):
                ctx = tuple(toks[i - self.prevctx_width : i])
                self.counts_base.add(ctx, toks[i], 1)

    def build_spoofer(
        self, 
        vocab_size: int, 
        spoofer_strength: float = 2.0,
    ) -> StaticSpoofer:
        """Build a static spoofer from learned statistics."""
        return StaticSpoofer(
            counts_base=self.counts_base,
            counts_wm=self.counts_wm,
            prevctx_width=self.prevctx_width,
            vocab_size=vocab_size,
            spoofer_strength=spoofer_strength,
        )
