"""
kgw_watermark.py — Standalone KGW-SelfHash watermark for GAN pipeline.

Implements KGW2-SelfHash (Kirchenbauer et al., 2024):
  - PRF seeded by: min{H(T_{t-h}), ..., H(T_{t-1}), H(T_t)} * ξ * H(T_t)
  - Vocabulary split into γ|V| green + (1-γ)|V| red tokens
  - Generation: logits[green] += δ
  - Detection: z = (n_green - γT) / sqrt(T * γ * (1-γ))

Designed for GAN adversarial training:
  - KGWOracle: generate watermarked/unwatermarked text
  - KGWDetector: compute z-score, detect watermark
  - KGWLogitsProcessor: HF-compatible logits processor
  - get_green_mask_for_sequence: token-level green/red annotation
"""

import torch
import torch.nn.functional as F
from math import sqrt
from typing import List, Tuple, Optional, Dict
from functools import lru_cache
from transformers import LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════
# Hash utilities
# ══════════════════════════════════════════════════════════════

# Global permutation table for hashing (deterministic)
_RNG = torch.Generator(device=torch.device("cpu"))
_RNG.manual_seed(2971215073)
_TABLE_SIZE = 1_000_003
_FIXED_TABLE = torch.randperm(_TABLE_SIZE, device=torch.device("cpu"), generator=_RNG)


def _hashint(x: torch.LongTensor) -> torch.LongTensor:
    """Hash integer tensor using fixed permutation table."""
    return _FIXED_TABLE[x.cpu() % _TABLE_SIZE] + 1


def _anchored_minhash_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    """
    SelfHash PRF: min{H(salt * T_i) * H(salt * T_anchor)} for all i.
    anchor = last token (self-seeding).
    """
    arr = salt_key * _hashint(input_ids) * _hashint(input_ids[-1])
    return arr.min().item()


# ══════════════════════════════════════════════════════════════
# KGW Core
# ══════════════════════════════════════════════════════════════

class KGWBase:
    """Base class for KGW watermark operations."""

    def __init__(
        self,
        vocab_size: int,
        gamma: float = 0.25,
        delta: float = 2.0,
        context_width: int = 4,  # h=3 + self-seeding → context_width=4
        hash_key: int = 15485863,
        device: str = "cuda",
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.context_width = context_width
        self.hash_key = hash_key
        self.device = device
        self.rng = torch.Generator(device=device)

    def _seed_rng(self, context_ids: torch.LongTensor) -> None:
        """Seed RNG from context tokens using anchored minhash PRF."""
        prf_key = _anchored_minhash_prf(context_ids[-self.context_width:], self.hash_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))

    def _get_greenlist_ids(self, context_ids: torch.LongTensor) -> torch.LongTensor:
        """Get green token IDs given context."""
        self._seed_rng(context_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        perm = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
        return perm[:greenlist_size]

    def is_green(self, context_ids: torch.LongTensor, token_id: int) -> bool:
        """Check if token_id is green given context."""
        greenlist = self._get_greenlist_ids(context_ids)
        return token_id in greenlist


# ══════════════════════════════════════════════════════════════
# Logits Processor (for HF generate)
# ══════════════════════════════════════════════════════════════

class KGWLogitsProcessor(LogitsProcessor, KGWBase):
    """HuggingFace LogitsProcessor that biases green tokens."""

    def __init__(self, vocab_size: int, **kwargs):
        KGWBase.__init__(self, vocab_size, **kwargs)
        self.self_salt = True  # SelfHash uses rejection sampling

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Add delta bias to green tokens. Handles self-seeding via rejection sampling."""
        if input_ids.shape[-1] < self.context_width:
            return scores

        for b in range(input_ids.shape[0]):
            # For self-seeding: iterate top-k candidates, check green with self included
            # Simplified: boost all tokens that would be green if they were selected
            # This is an approximation — full rejection sampling is expensive
            context = input_ids[b]
            greenlist = self._get_greenlist_ids_selfhash(context, scores[b])
            mask = torch.zeros_like(scores[b], dtype=torch.bool)
            mask[greenlist] = True
            scores[b][mask] += self.delta

        return scores

    def _get_greenlist_ids_selfhash(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.LongTensor:
        """Get green list with self-hashing (token itself in PRF seed)."""
        # Get top-k candidates
        k = min(50, self.vocab_size)
        topk_ids = torch.topk(scores, k, largest=True, sorted=False).indices

        greenlist = []
        for cand in topk_ids:
            # Context = [...previous tokens, candidate_token]
            context_with_self = torch.cat([input_ids, cand.unsqueeze(0)])
            if context_with_self.shape[0] >= self.context_width:
                gl = self._get_greenlist_ids(context_with_self)
                if cand.item() in gl:
                    greenlist.append(cand.item())

        return torch.tensor(greenlist, device=self.device, dtype=torch.long)


# ══════════════════════════════════════════════════════════════
# KGW Detector (z-score based)
# ══════════════════════════════════════════════════════════════

class KGWDetector(KGWBase):
    """Detect KGW watermark via z-score computation."""

    def __init__(
        self,
        vocab_size: int,
        z_threshold: float = 4.0,
        ignore_repeated_ngrams: bool = True,
        **kwargs,
    ):
        super().__init__(vocab_size, **kwargs)
        self.z_threshold = z_threshold
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def detect(self, token_ids: torch.LongTensor) -> Dict[str, float]:
        """
        Detect watermark in token sequence.

        Returns:
            Dict with z_score, p_value, green_fraction, is_watermarked, num_green, num_scored
        """
        green_mask, green_count, total_scored = self._score_sequence(token_ids)
        z_score = self._compute_z_score(green_count, total_scored)
        p_value = self._compute_p_value(z_score)

        return {
            "z_score": z_score,
            "p_value": p_value,
            "green_fraction": green_count / max(total_scored, 1),
            "is_watermarked": z_score > self.z_threshold,
            "num_green": green_count,
            "num_scored": total_scored,
        }

    def detect_text(self, text: str, tokenizer) -> Dict[str, float]:
        """Detect watermark from text string."""
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        return self.detect(ids)

    def get_z_score(self, token_ids: torch.LongTensor) -> float:
        """Quick z-score computation."""
        _, green_count, total_scored = self._score_sequence(token_ids)
        return self._compute_z_score(green_count, total_scored)

    def _score_sequence(
        self, token_ids: torch.LongTensor
    ) -> Tuple[List[bool], int, int]:
        """Score each token as green/red."""
        token_ids = token_ids.cpu()
        green_mask = []
        green_count = 0
        seen_ngrams = set()

        for idx in range(self.context_width, len(token_ids)):
            # Context includes the token itself (self-seeding)
            context = token_ids[:idx + 1]

            # Check for repeated ngrams
            if self.ignore_repeated_ngrams:
                ngram = tuple(token_ids[idx - self.context_width: idx + 1].tolist())
                if ngram in seen_ngrams:
                    continue
                seen_ngrams.add(ngram)

            # Check if token is green
            greenlist = self._get_greenlist_ids(context.to(self.device))
            is_green = token_ids[idx].item() in greenlist.cpu()
            green_mask.append(is_green)
            if is_green:
                green_count += 1

        total_scored = len(green_mask)
        return green_mask, green_count, total_scored

    def _compute_z_score(self, green_count: int, T: int) -> float:
        if T == 0:
            return 0.0
        numer = green_count - self.gamma * T
        denom = sqrt(T * self.gamma * (1 - self.gamma))
        return numer / denom if denom > 0 else 0.0

    def _compute_p_value(self, z: float) -> float:
        import scipy.stats
        return scipy.stats.norm.sf(z)


# ══════════════════════════════════════════════════════════════
# KGW Oracle (generate watermarked/unwatermarked text)
# ══════════════════════════════════════════════════════════════

class KGWOracle:
    """
    KGW watermark Oracle — generates watermarked text and detects.
    Replaces UPVOracle in the GAN pipeline.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-1.3b",
        device: str = "cuda",
        gamma: float = 0.25,
        delta: float = 2.0,
        context_width: int = 4,
        hash_key: int = 15485863,
        z_threshold: float = 4.0,
        max_new_tokens: int = 256,
    ):
        self.device = device
        self.gamma = gamma
        self.delta = delta
        self.context_width = context_width
        self.hash_key = hash_key

        # Load LLM
        print(f"[KGWOracle] Loading LLM: {model_name}")
        load_kwargs = {"torch_dtype": torch.float16 if "cuda" in device else torch.float32}
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("[KGWOracle] Using flash_attention_2")
        except ImportError:
            print("[KGWOracle] Using default attention")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.vocab_size = self.model.config.vocab_size
        self.max_new_tokens = max_new_tokens

        # Build logits processor
        self.logits_processor = KGWLogitsProcessor(
            vocab_size=self.vocab_size,
            gamma=gamma, delta=delta,
            context_width=context_width,
            hash_key=hash_key,
            device=device,
        )

        # Build detector
        self.detector = KGWDetector(
            vocab_size=self.vocab_size,
            gamma=gamma, delta=delta,
            context_width=context_width,
            hash_key=hash_key,
            z_threshold=z_threshold,
            device=device,
        )

        print(f"[KGWOracle] Ready. gamma={gamma}, delta={delta}, h={context_width-1} (context_width={context_width})")

    @torch.no_grad()
    def generate_watermarked(
        self, prompts: List[str], max_new_tokens: int = None
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate watermarked text using KGW logits processor."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        texts, ids_list = [], []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                logits_processor=LogitsProcessorList([self.logits_processor]),
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            texts.append(text)
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            ids_list.append(gen_ids.cpu())

        return texts, ids_list

    @torch.no_grad()
    def generate_unwatermarked(
        self, prompts: List[str], max_new_tokens: int = None
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate text WITHOUT watermark."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        texts, ids_list = [], []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            texts.append(text)
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            ids_list.append(gen_ids.cpu())

        return texts, ids_list

    def detect_watermark(self, text: str) -> Dict[str, float]:
        """Detect watermark in text. Returns dict with z_score, is_watermarked, etc."""
        return self.detector.detect_text(text, self.tokenizer)
