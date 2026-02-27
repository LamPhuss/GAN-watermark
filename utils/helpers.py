# ============================================================
# helpers.py
# Description: Utility functions for watermark GAN
#   FIX 5: Added window_size, layers, pretrain_data_mode fields.
# ============================================================

import os
import json
import yaml
import random
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class GANConfig:
    """Parsed GAN configuration."""
    # Model
    llm_name: str
    device: str
    upv_config_path: str
    upv_generator_weights: str
    upv_detector_weights: str

    # Watermark
    gamma: float
    delta: float
    z_threshold: float
    prefix_length: int
    window_size: int            # FIX 5: NEW
    layers: int                 # FIX 5: NEW (b_layers for SubNet)
    bit_number: int
    sigma: float
    default_top_k: int

    # Discriminator (UPV Detector)
    disc_bit_number: int
    disc_freeze_embedding: bool
    disc_pretrain_epochs: int
    disc_pretrain_lr: float
    disc_pretrain_batch_size: int
    disc_pretrain_num_samples: int
    disc_pretrain_data_mode: str  # FIX 3: "random_tokens" or "llm_text"

    # Attacker
    att_lora_r: int
    att_lora_alpha: int
    att_lora_dropout: float
    att_lora_target_modules: list
    att_pretrain_epochs: int
    att_pretrain_lr: float
    att_pretrain_batch_size: int
    att_pretrain_num_samples: int
    att_pretrain_max_length: int
    att_learning_mode: str
    att_learning_num_queries: int
    att_prevctx_width: int

    # Monte Carlo Search
    mc_num_chunks: int
    mc_num_rollouts: int
    mc_batch_size: int
    mc_rollout_policy: str

    # Adversarial
    adv_num_epochs: int
    adv_max_gen_length: int
    adv_d_steps: int
    adv_d_lr: float
    adv_g_steps: int
    adv_g_lr: float
    adv_lambda_reward: float
    adv_lambda_ppl: float
    adv_reward_baseline: float
    adv_temperature: float
    adv_eval_every: int
    adv_checkpoint_every: int
    adv_checkpoint_dir: str
    adv_d_label_smoothing: float    # Label smoothing for D (real target = 1 - smooth)
    adv_diversity_reward: float     # Bonus weight for diverse generation
    # Data
    dataset_path: str
    max_prompt_length: int
    num_prompts: int


def load_config(config_path: str) -> GANConfig:
    """Load and parse the YAML config file into GANConfig."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return GANConfig(
        # Model
        llm_name=cfg['model']['llm_name'],
        device=cfg['model']['device'],
        upv_config_path=cfg['model']['upv_config_path'],
        upv_generator_weights=cfg['model']['upv_generator_weights'],
        upv_detector_weights=cfg['model']['upv_detector_weights'],

        # Watermark
        gamma=cfg['watermark']['gamma'],
        delta=cfg['watermark']['delta'],
        z_threshold=cfg['watermark']['z_threshold'],
        prefix_length=cfg['watermark']['prefix_length'],
        window_size=cfg['watermark']['window_size'],          # FIX 5
        layers=cfg['watermark']['layers'],                    # FIX 5
        bit_number=cfg['watermark']['bit_number'],
        sigma=cfg['watermark']['sigma'],
        default_top_k=cfg['watermark']['default_top_k'],

        # Discriminator
        disc_bit_number=cfg['discriminator']['bit_number'],
        disc_freeze_embedding=cfg['discriminator']['freeze_embedding'],
        disc_pretrain_epochs=cfg['discriminator']['pretrain_epochs'],
        disc_pretrain_lr=cfg['discriminator']['pretrain_lr'],
        disc_pretrain_batch_size=cfg['discriminator']['pretrain_batch_size'],
        disc_pretrain_num_samples=cfg['discriminator']['pretrain_num_samples'],
        disc_pretrain_data_mode=cfg['discriminator'].get('pretrain_data_mode', 'random_tokens'),

        # Attacker
        att_lora_r=cfg['attacker']['lora_r'],
        att_lora_alpha=cfg['attacker']['lora_alpha'],
        att_lora_dropout=cfg['attacker']['lora_dropout'],
        att_lora_target_modules=cfg['attacker']['lora_target_modules'],
        att_pretrain_epochs=cfg['attacker']['pretrain_epochs'],
        att_pretrain_lr=cfg['attacker']['pretrain_lr'],
        att_pretrain_batch_size=cfg['attacker']['pretrain_batch_size'],
        att_pretrain_num_samples=cfg['attacker']['pretrain_num_samples'],
        att_pretrain_max_length=cfg['attacker']['pretrain_max_length'],
        att_learning_mode=cfg['attacker']['learning_mode'],
        att_learning_num_queries=cfg['attacker']['learning_num_queries'],
        att_prevctx_width=cfg['attacker']['prevctx_width'],

        # Monte Carlo Search
        mc_num_chunks=cfg['monte_carlo']['num_chunks'],
        mc_num_rollouts=cfg['monte_carlo']['num_rollouts'],
        mc_batch_size=cfg['monte_carlo']['batch_size'],
        mc_rollout_policy=cfg['monte_carlo']['rollout_policy'],

        # Adversarial
        adv_num_epochs=cfg['adversarial']['num_epochs'],
        adv_max_gen_length=cfg['adversarial']['max_gen_length'],
        adv_d_steps=cfg['adversarial']['d_steps_per_epoch'],
        adv_d_lr=cfg['adversarial']['d_lr'],
        adv_g_steps=cfg['adversarial']['g_steps_per_epoch'],
        adv_g_lr=cfg['adversarial']['g_lr'],
        adv_lambda_reward=cfg['adversarial']['lambda_reward'],
        adv_lambda_ppl=cfg['adversarial']['lambda_ppl'],
        adv_reward_baseline=cfg['adversarial']['reward_baseline'],
        adv_temperature=cfg['adversarial']['temperature'],
        adv_eval_every=cfg['adversarial']['eval_every'],
        adv_checkpoint_every=cfg['adversarial']['checkpoint_every'],
        adv_checkpoint_dir=cfg['adversarial']['checkpoint_dir'],
        adv_d_label_smoothing=cfg['adversarial'].get('d_label_smoothing', 0.0),
        adv_diversity_reward=cfg['adversarial'].get('diversity_reward_weight', 0.0),

        # Data
        dataset_path=cfg['data']['dataset_path'],
        max_prompt_length=cfg['data']['max_prompt_length'],
        num_prompts=cfg['data']['num_prompts'],
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_metrics(metrics: Dict[str, float], step: int) -> None:
    """Print metrics in a formatted way."""
    parts = [f"[Step {step:04d}]"]
    for key, value in metrics.items():
        parts.append(f"{key}: {value:.4f}")
    print(" | ".join(parts))


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def compute_ppl_from_logprobs(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from log probabilities."""
    masked_log_probs = log_probs * mask
    counts = mask.sum(dim=-1).clamp(min=1)
    avg_nll = -masked_log_probs.sum(dim=-1) / counts
    return torch.exp(avg_nll)


def compute_diversity_score(token_ids_list: List[List[int]]) -> float:
    """
    Compute diversity score based on unique token ratio.
    Low diversity suggests mode collapse.
    """
    if not token_ids_list:
        return 0.0

    all_tokens = []
    for ids in token_ids_list:
        all_tokens.extend(ids)

    if len(all_tokens) == 0:
        return 0.0

    unique_ratio = len(set(all_tokens)) / len(all_tokens)
    return unique_ratio
