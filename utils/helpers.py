# ============================================================
# helpers.py — KGW version
# ============================================================

import os, json, yaml, random, torch
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class GANConfig:
    """Parsed GAN configuration — KGW version."""
    # Model
    llm_name: str
    device: str

    # KGW Watermark
    wm_gamma: float
    wm_delta: float
    wm_context_width: int
    wm_hash_key: int
    wm_z_threshold: float

    # Discriminator (z-score based)
    disc_mode: str
    disc_z_center: float
    disc_temperature: float

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
    att_spoofer_strength: float

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
    adv_d_label_smoothing: float
    adv_diversity_reward: float

    # Data
    dataset_path: str
    max_prompt_length: int
    num_prompts: int


def load_config(config_path: str) -> GANConfig:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return GANConfig(
        llm_name=cfg['model']['llm_name'],
        device=cfg['model']['device'],

        wm_gamma=cfg['watermark']['gamma'],
        wm_delta=cfg['watermark']['delta'],
        wm_context_width=cfg['watermark']['context_width'],
        wm_hash_key=cfg['watermark']['hash_key'],
        wm_z_threshold=cfg['watermark']['z_threshold'],

        disc_mode=cfg['discriminator']['mode'],
        disc_z_center=cfg['discriminator']['z_center'],
        disc_temperature=cfg['discriminator']['temperature'],

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
        att_spoofer_strength=cfg['attacker'].get('spoofer_strength', 7.5),

        mc_num_chunks=cfg['monte_carlo']['num_chunks'],
        mc_num_rollouts=cfg['monte_carlo']['num_rollouts'],
        mc_batch_size=cfg['monte_carlo']['batch_size'],
        mc_rollout_policy=cfg['monte_carlo']['rollout_policy'],

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
        adv_diversity_reward=cfg['adversarial'].get('diversity_reward_weight', 0.05),

        dataset_path=cfg['data']['dataset_path'],
        max_prompt_length=cfg['data']['max_prompt_length'],
        num_prompts=cfg['data']['num_prompts'],
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log_metrics(metrics: Dict[str, float], step: int):
    parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    print(f"[Step {step:04d}] " + " | ".join(parts))


def compute_ppl_from_logprobs(log_probs, mask=None):
    if mask is not None:
        log_probs = log_probs * mask
        avg = log_probs.sum() / mask.sum().clamp(min=1)
    else:
        avg = log_probs.mean()
    return torch.exp(-avg).item()


def compute_diversity_score(token_id_lists):
    if not token_id_lists:
        return 0.0
    all_toks = []
    for ids in token_id_lists:
        all_toks.extend(ids)
    if len(all_toks) == 0:
        return 0.0
    return len(set(all_toks)) / len(all_toks)
