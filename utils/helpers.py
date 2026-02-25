# ============================================================
# helpers.py
# Description: Utility functions for watermark GAN
#   Updated with Monte Carlo Search config fields.
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
        bit_number=cfg['watermark']['bit_number'],
        sigma=cfg['watermark']['sigma'],
        default_top_k=cfg['watermark']['default_top_k'],
        
        # Discriminator (UPV Detector)
        disc_bit_number=cfg['discriminator']['bit_number'],
        disc_freeze_embedding=cfg['discriminator']['freeze_embedding'],
        disc_pretrain_epochs=cfg['discriminator']['pretrain_epochs'],
        disc_pretrain_lr=cfg['discriminator']['pretrain_lr'],
        disc_pretrain_batch_size=cfg['discriminator']['pretrain_batch_size'],
        disc_pretrain_num_samples=cfg['discriminator']['pretrain_num_samples'],
        
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
        
        # Data
        dataset_path=cfg['data']['dataset_path'],
        max_prompt_length=cfg['data']['max_prompt_length'],
        num_prompts=cfg['data']['num_prompts'],
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_metrics(metrics: Dict[str, float], epoch: int, log_file: Optional[str] = None) -> None:
    msg = f"[Epoch {epoch:04d}] " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(msg)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg + "\n")


def compute_ppl_from_logprobs(log_probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    total_log_prob = log_probs.sum(dim=-1)
    avg_log_prob = total_log_prob / lengths.float()
    return torch.exp(-avg_log_prob)


def compute_diversity_score(token_ids_batch: list) -> float:
    all_tokens = []
    for seq in token_ids_batch:
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        all_tokens.extend(seq)
    if len(all_tokens) == 0:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)
