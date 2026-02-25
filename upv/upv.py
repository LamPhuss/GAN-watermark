"""
upv.py  (ported from MarkLLM/watermark/upv/upv.py — standalone)
Description: Self-contained UPV watermark algorithm.

All MarkLLM-specific imports have been eliminated:
  - BaseConfig / BaseWatermark   → replaced by plain Python classes
  - load_config_file             → replaced by inline json.load
  - TransformersConfig           → imported from local upv/transformers_config.py
  - DataForVisualization         → removed (not needed for GAN training)
  - AlgorithmNameMismatchError   → replaced by plain ValueError

Logic is identical to MarkLLM's implementation.
"""

import json
import torch
from math import sqrt
from functools import partial
from typing import Optional

from .network_model import UPVGenerator, UPVDetector
from .transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList


# ──────────────────────────────────────────────────────────────
# Helpers (replaces MarkLLM's utils/utils.py)
# ──────────────────────────────────────────────────────────────

def _load_config_file(path: str) -> dict:
    """Load a JSON config file and return as dict. Raises on error."""
    with open(path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# UPVConfig  (replaces BaseConfig + UPVConfig from MarkLLM)
# ──────────────────────────────────────────────────────────────

class UPVConfig:
    """
    Configuration for the UPV watermark algorithm.
    Loads parameters from a JSON file and model info from TransformersConfig.
    """

    algorithm_name = "UPV"

    def __init__(
        self,
        algorithm_config_path: Optional[str],
        transformers_config: TransformersConfig,
        **kwargs,
    ) -> None:
        # Load JSON config
        if algorithm_config_path is None:
            algorithm_config_path = "upv/UPV.json"
        config_dict = _load_config_file(algorithm_config_path)
        config_dict.update(kwargs)  # allow overrides

        # Validate algorithm name
        if config_dict.get("algorithm_name", "UPV") != "UPV":
            raise ValueError(
                f"Config algorithm_name mismatch: expected 'UPV', "
                f"got '{config_dict.get('algorithm_name')}'"
            )

        # Watermark parameters
        self.gamma            = config_dict["gamma"]
        self.delta            = config_dict["delta"]
        self.z_threshold      = config_dict["z_threshold"]
        self.prefix_length    = config_dict["prefix_length"]
        self.bit_number       = config_dict["bit_number"]
        self.sigma            = config_dict["sigma"]
        self.default_top_k    = config_dict["default_top_k"]
        self.generator_model_name = config_dict["generator_model_name"]
        self.detector_model_name  = config_dict["detector_model_name"]
        self.detect_mode      = config_dict["detect_mode"]

        # Model / tokenizer from TransformersConfig
        self.generation_model     = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size           = transformers_config.vocab_size
        self.device               = transformers_config.device
        self.gen_kwargs           = transformers_config.gen_kwargs


# ──────────────────────────────────────────────────────────────
# UPVUtils
# ──────────────────────────────────────────────────────────────

class UPVUtils:
    """Utility / helper class for UPV — greenlist prediction and scoring."""

    def __init__(self, config: UPVConfig) -> None:
        self.config = config
        self.generator_model = self._load_generator().to(config.device)
        self.detector_model  = self._load_detector().to(config.device)
        self.cache: dict = {}
        self.top_k     = config.gen_kwargs.get("top_k",    config.default_top_k)
        self.num_beams = config.gen_kwargs.get("num_beams", None)

    def _load_generator(self) -> UPVGenerator:
        model = UPVGenerator(self.config.bit_number, self.config.prefix_length + 1)
        state = torch.load(self.config.generator_model_name, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def _load_detector(self) -> UPVDetector:
        model = UPVDetector(self.config.bit_number)
        state = torch.load(self.config.detector_model_name, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    # ── helpers ──

    def int_to_bin_list(self, n: int, length: int = 8) -> list:
        """Convert integer to a binary list of given length."""
        bin_str = format(n, "b")[:length].zfill(length)
        return [int(b) for b in bin_str]

    @torch.no_grad()
    def _get_predictions_from_generator(self, input_x: torch.Tensor) -> bool:
        out = self.generator_model(input_x)
        return (out > 0.5).bool().item()

    def _select_candidates(self, scores: torch.Tensor) -> torch.Tensor:
        if self.num_beams is not None:
            threshold = torch.topk(scores, self.num_beams, largest=True, sorted=False)[0][-1]
            return (scores >= (threshold - self.config.delta)).nonzero(as_tuple=True)[0]
        return torch.topk(scores, self.top_k, largest=True, sorted=False).indices

    # ── greenlist ──

    def get_greenlist_ids(self, input_ids: torch.Tensor, scores: torch.Tensor) -> list:
        """Return list of 'green' candidate token IDs for generation."""
        candidates = self._select_candidates(scores)
        ids_list   = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

        greenlist = []
        for v in candidates:
            pair  = ids_list[-self.config.prefix_length:] + [v.item()] \
                    if self.config.prefix_length > 0 else [v.item()]
            key   = tuple(pair)
            if key in self.cache:
                result = self.cache[key]
            else:
                bin_list = [self.int_to_bin_list(num, self.config.bit_number) for num in pair]
                tensor   = torch.tensor(bin_list, device=self.config.device).float().unsqueeze(0)
                result   = self._get_predictions_from_generator(tensor)
                self.cache[key] = result
            if result:
                greenlist.append(int(v))
        return greenlist

    def _judge_green(self, input_ids: torch.Tensor, current_token: int) -> bool:
        """Check if `current_token` is green given previous context."""
        last = list(input_ids[-self.config.prefix_length:]) \
               if self.config.prefix_length > 0 else []
        pair = last + [current_token]
        key  = tuple(pair)
        if key in self.cache:
            return self.cache[key]
        bin_list = [self.int_to_bin_list(n, self.config.bit_number) for n in pair]
        tensor   = torch.tensor(bin_list, device=self.config.device).float().unsqueeze(0)
        result   = self._get_predictions_from_generator(tensor)
        self.cache[key] = result
        return result

    # ── scoring ──

    def green_token_mask_and_stats(
        self, input_ids: torch.Tensor
    ) -> tuple[list, int, float]:
        """Return (mask, green_count, z_score) for a token sequence."""
        mask  = [None] * self.config.prefix_length
        green = 0
        for idx in range(self.config.prefix_length, len(input_ids)):
            if self._judge_green(input_ids[:idx], int(input_ids[idx])):
                mask.append(True)
                green += 1
            else:
                mask.append(False)
        scored = len(input_ids) - self.config.prefix_length
        z      = self._compute_z_score(green, scored)
        return mask, green, z

    def _compute_z_score(self, observed: int, T: int) -> float:
        exp  = self.config.gamma
        num  = observed - exp * T
        denom = sqrt(T * exp * (1 - exp) + self.config.sigma ** 2 * T)
        return num / denom if denom != 0 else 0.0


# ──────────────────────────────────────────────────────────────
# UPVLogitsProcessor
# ──────────────────────────────────────────────────────────────

class UPVLogitsProcessor(LogitsProcessor):
    """HuggingFace LogitsProcessor that biases green tokens during generation."""

    def __init__(self, config: UPVConfig, utils: UPVUtils) -> None:
        self.config = config
        self.utils  = utils

    def _bias_greenlist_logits(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        bias: float,
    ) -> torch.Tensor:
        scores[mask] = scores[mask] + bias
        return scores

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        mask = torch.zeros_like(scores)
        for b in range(input_ids.shape[0]):
            gids = self.utils.get_greenlist_ids(input_ids[b], scores=scores[b])
            mask[b][gids] = 1
        scores = self._bias_greenlist_logits(scores, mask.bool(), self.config.delta)
        return scores


# ──────────────────────────────────────────────────────────────
# UPV  (top-level class)
# ──────────────────────────────────────────────────────────────

class UPV:
    """
    Top-level UPV watermark class.
    Provides generate_watermarked_text(), generate_unwatermarked_text(), detect_watermark().
    """

    def __init__(
        self,
        algorithm_config_path: Optional[str],
        transformers_config: TransformersConfig,
        **kwargs,
    ) -> None:
        self.config          = UPVConfig(algorithm_config_path, transformers_config, **kwargs)
        self.utils           = UPVUtils(self.config)
        self.logits_processor = UPVLogitsProcessor(self.config, self.utils)

    # ── generation ──

    def generate_watermarked_text(self, prompt: str) -> str:
        generate = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )
        enc = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        out = generate(**enc)
        return self.config.generation_tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def generate_unwatermarked_text(self, prompt: str) -> str:
        enc = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        out = self.config.generation_model.generate(**enc, **self.config.gen_kwargs)
        return self.config.generation_tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    # ── detection ──

    def _detect_network_mode(self, enc: torch.Tensor) -> tuple[bool, None]:
        bins = [self.utils.int_to_bin_list(t.item(), self.config.bit_number) for t in enc]
        x    = torch.tensor(bins, device=self.config.device).float().unsqueeze(0)
        out  = self.utils.detector_model(x).reshape(-1)
        return bool((out.data > 0.5).int().sum().item() > 0), None

    def detect_watermark(self, text: str, return_dict: bool = True):
        enc = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        if self.config.detect_mode == "key":
            _, _, z_score   = self.utils.green_token_mask_and_stats(enc)
            is_watermarked  = z_score > self.config.z_threshold
        else:
            is_watermarked, z_score = self._detect_network_mode(enc)

        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        return (is_watermarked, z_score)
