# ============================================================
# data_generator.py  (standalone — zero MarkLLM dependency)
# Description: Generate training data using UPV watermark.
#   + Flash Attention 2 for faster LLM inference.
# ============================================================

import os
import json
import torch
from typing import List, Tuple, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from upv.upv import UPV
from upv.transformers_config import TransformersConfig


def _load_model_with_flash_attn(model_name: str, device: str) -> AutoModelForCausalLM:
    """
    Load HuggingFace causal LM with Flash Attention 2 if available.
    Falls back to default attention if flash_attn not installed.
    """
    load_kwargs = {
        "torch_dtype": torch.float16 if "cuda" in device else torch.float32,
    }
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"[FlashAttn] Oracle: using flash_attention_2")
    except ImportError:
        print(f"[FlashAttn] Oracle: flash_attn not found, using default")

    return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)


class UPVOracle:
    """
    UPV Generator (Oracle / Ground Truth).
    Weights are FROZEN — only generates real watermarked text.
    Uses Flash Attention 2 for faster generation.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-1.3b",
        device: str = "cuda",
        upv_config_path: str = "upv/UPV.json",
        max_new_tokens: int = 256,
        do_sample: bool = True,
        no_repeat_ngram_size: int = 4,
    ):
        self.device = device

        print(f"[UPVOracle] Loading LLM: {model_name}")
        self.model = _load_model_with_flash_attn(model_name, device)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze LLM weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        transformers_config = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_size=self.model.config.vocab_size,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        print(f"[UPVOracle] Loading UPV watermark from: {upv_config_path}")
        self.watermark = UPV(
            algorithm_config_path=upv_config_path,
            transformers_config=transformers_config,
        )

        print("[UPVOracle] Ready. All weights FROZEN.")

    @torch.no_grad()
    def generate_watermarked(
        self, prompts: List[str], max_new_tokens: int = 256
    ) -> Tuple[List[str], List[torch.Tensor]]:
        texts, ids_list = [], []
        for prompt in prompts:
            try:
                text = self.watermark.generate_watermarked_text(prompt)
                texts.append(text)
                ids = self.tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                ids_list.append(ids)
            except Exception as e:
                print(f"[UPVOracle] generate_watermarked error: {e}")
                texts.append("")
                ids_list.append(torch.tensor([]))
        return texts, ids_list

    @torch.no_grad()
    def generate_unwatermarked(
        self, prompts: List[str], max_new_tokens: int = 256
    ) -> Tuple[List[str], List[torch.Tensor]]:
        texts, ids_list = [], []
        for prompt in prompts:
            try:
                text = self.watermark.generate_unwatermarked_text(prompt)
                texts.append(text)
                ids = self.tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                ids_list.append(ids)
            except Exception as e:
                print(f"[UPVOracle] generate_unwatermarked error: {e}")
                texts.append("")
                ids_list.append(torch.tensor([]))
        return texts, ids_list

    def detect_watermark(self, text: str) -> dict:
        return self.watermark.detect_watermark(text)


class DataGenerator:
    """Generate pre-training datasets for Discriminator and Attacker."""

    _FALLBACK_PROMPTS = [
        "The latest research in artificial intelligence suggests that",
        "In a groundbreaking study, scientists discovered that",
        "The economic impact of climate change has been assessed by",
        "According to recent findings in neuroscience,",
        "The development of quantum computing has led to",
        "Historians have long debated the true causes of",
        "New regulations aimed at reducing carbon emissions will",
        "The relationship between diet and mental health is",
    ] * 200

    def __init__(self, oracle: UPVOracle, dataset_path: str):
        self.oracle = oracle
        self.prompts = self._load_prompts(dataset_path)

    def _load_prompts(self, path: str) -> List[str]:
        prompts = []
        try:
            with open(path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    if "prompt" in item:
                        prompts.append(item["prompt"])
            print(f"[DataGenerator] Loaded {len(prompts)} prompts from {path}")
        except FileNotFoundError:
            print(f"[DataGenerator] Dataset not found at '{path}'. Using fallback.")
            prompts = self._FALLBACK_PROMPTS.copy()
        return prompts

    def generate_discriminator_data(
        self, num_samples: int = 10_000, batch_size: int = 8,
        save_path: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        real_texts, natural_texts = [], []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            lo = i * batch_size
            hi = min(lo + batch_size, num_samples)
            batch = [self.prompts[j % len(self.prompts)] for j in range(lo, hi)]

            wm, _ = self.oracle.generate_watermarked(batch)
            nat, _ = self.oracle.generate_unwatermarked(batch)
            real_texts.extend(wm)
            natural_texts.extend(nat)

            if (i + 1) % 50 == 0:
                print(f"[DataGenerator] D-data: {len(real_texts)}/{num_samples}")

        real_texts = real_texts[:num_samples]
        natural_texts = natural_texts[:num_samples]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({"real_watermarked": real_texts, "natural": natural_texts}, f)
            print(f"[DataGenerator] Saved D-data to {save_path}")

        return real_texts, natural_texts

    def generate_attacker_sft_data(
        self, static_spoofer, attacker_llm,
        num_samples: int = 10_000, batch_size: int = 8,
        save_path: Optional[str] = None,
    ) -> List[dict]:
        sft_data = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            lo = i * batch_size
            hi = min(lo + batch_size, num_samples)
            batch = [self.prompts[j % len(self.prompts)] for j in range(lo, hi)]

            texts, _ = attacker_llm.generate(
                batch, max_length=256, temperature=1.0,
                static_spoofer=static_spoofer,
            )

            for prompt, text in zip(batch, texts):
                sft_data.append({
                    "prompt": prompt,
                    "response": text.replace(prompt, "").strip(),
                })

            if (i + 1) % 50 == 0:
                print(f"[DataGenerator] SFT data: {len(sft_data)}/{num_samples}")

        sft_data = sft_data[:num_samples]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(sft_data, f)
            print(f"[DataGenerator] Saved SFT data to {save_path}")

        return sft_data


def pad_sequences(
    token_ids_list: List[torch.Tensor],
    pad_value: int = 0,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [len(ids) for ids in token_ids_list]
    if max_len is None:
        max_len = max(lengths) if lengths else 1

    padded = torch.full((len(token_ids_list), max_len), pad_value, dtype=torch.long)
    for i, ids in enumerate(token_ids_list):
        end = min(len(ids), max_len)
        padded[i, :end] = ids[:end]

    return padded, torch.tensor(lengths, dtype=torch.long)
