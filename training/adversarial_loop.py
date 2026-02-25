# ============================================================
# adversarial_loop.py
# Description: Step 3 — The GAN Adversarial Training Loop
#   Attacker (G_θ) vs Discriminator (D_φ)
# ============================================================

import os
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from models.detector import WatermarkDiscriminator
from models.attacker import AttackerLLM, StaticSpoofer
from data.data_generator import UPVOracle, pad_sequences
from utils.helpers import (
    GANConfig, set_seed, log_metrics, ensure_dir, 
    compute_ppl_from_logprobs, compute_diversity_score
)


class AdversarialTrainer:
    """
    Implements the GAN Adversarial Loop (PHẦN 3 in the design doc).
    
    The Minimax Game:
        - D_φ maximizes: E[log D(real)] + E[log(1 - D(fake))]
        - G_θ maximizes: E[D(fake)] (via REINFORCE policy gradient)
    
    Each epoch:
        1. G generates fake watermarked text
        2. Oracle generates real watermarked text (same prompts)
        3. D is updated on (real=1, fake=0)
        4. G is updated via REINFORCE (reward from updated D)
    """

    def __init__(
        self,
        config: GANConfig,
        discriminator: WatermarkDiscriminator,
        attacker: AttackerLLM,
        oracle: UPVOracle,
        static_spoofer: Optional[StaticSpoofer] = None,
    ):
        self.config = config
        self.device = config.device
        
        # Models
        self.discriminator = discriminator
        self.attacker = attacker
        self.oracle = oracle
        self.static_spoofer = static_spoofer
        
        # Optimizers
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.get_trainable_params(),
            lr=config.adv_d_lr,
        )
        self.g_optimizer = torch.optim.Adam(
            self.attacker.get_lora_params(),
            lr=config.adv_g_lr,
        )
        
        # Loss
        self.bce_loss = nn.BCELoss()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'd_loss': [],
            'g_loss': [],
            'd_reward_real': [],
            'd_reward_fake': [],
            'spoofing_rate': [],
            'avg_ppl': [],
            'diversity': [],
        }
        
        print(f"[AdversarialTrainer] Initialized with {len(self.prompts)} prompts")

    def _load_prompts(self) -> List[str]:
        """Load prompts from dataset."""
        prompts = []
        try:
            with open(self.config.dataset_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'prompt' in item:
                        prompts.append(item['prompt'])
            prompts = prompts[:self.config.num_prompts]
        except:
            prompts = [
                "The latest research in artificial intelligence suggests that",
                "In a groundbreaking study, scientists discovered that",
                "The economic impact of climate change has been",
            ] * 100
        return prompts

    def _get_batch_prompts(self, epoch: int) -> List[str]:
        """Get a batch of prompts for the current epoch."""
        batch_size = self.config.adv_batch_size
        start = (epoch * batch_size) % len(self.prompts)
        
        batch = []
        for i in range(batch_size):
            idx = (start + i) % len(self.prompts)
            batch.append(self.prompts[idx])
        return batch

    # ────────────────────────────────────────────────────────────
    # STEP 3.1: Generation Phase
    # ────────────────────────────────────────────────────────────

    def generation_phase(self, prompts: List[str]) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor]:
        """
        Step 3.1: Attacker generates fake data, Oracle generates real data.
        
        Args:
            prompts: batch of prompts
        
        Returns:
            (fake_texts, real_texts, fake_token_ids, real_token_ids)
        """
        # Fake: Attacker generates
        fake_texts, fake_ids = self.attacker.generate(
            prompts,
            max_length=self.config.adv_max_gen_length,
            temperature=self.config.adv_temperature,
            do_sample=True,
            static_spoofer=self.static_spoofer,
        )
        
        # Real: Oracle generates (frozen UPV Generator)
        real_texts, real_ids_list = self.oracle.generate_watermarked(
            prompts,
            max_length=self.config.adv_max_gen_length,
        )
        
        # Tokenize real texts using attacker's tokenizer (for D compatibility)
        real_token_ids_list = []
        for text in real_texts:
            ids = self.attacker.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=self.config.adv_max_gen_length, 
                add_special_tokens=False,
            )["input_ids"][0]
            real_token_ids_list.append(ids)
        
        # Pad both to same format
        fake_token_ids_list = []
        for i in range(len(fake_texts)):
            ids = self.attacker.tokenizer(
                fake_texts[i], return_tensors="pt", truncation=True,
                max_length=self.config.adv_max_gen_length,
                add_special_tokens=False,
            )["input_ids"][0]
            fake_token_ids_list.append(ids)
        
        # Pad
        pad_id = self.attacker.tokenizer.pad_token_id or 0
        fake_padded, fake_lengths = pad_sequences(fake_token_ids_list, pad_value=pad_id)
        real_padded, real_lengths = pad_sequences(real_token_ids_list, pad_value=pad_id)
        
        return (
            fake_texts, real_texts,
            fake_padded.to(self.device), real_padded.to(self.device),
            fake_lengths.to(self.device), real_lengths.to(self.device),
        )

    # ────────────────────────────────────────────────────────────
    # STEP 3.2: Discriminator Update
    # ────────────────────────────────────────────────────────────

    def update_discriminator(
        self,
        fake_ids: torch.Tensor, fake_lengths: torch.Tensor,
        real_ids: torch.Tensor, real_lengths: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Step 3.2: Update Discriminator (D_φ).
        
        D is trained to output:
            - 1 for real watermarked text
            - 0 for fake (attacker) text
        
        Loss = BCE(D(real), 1) + BCE(D(fake), 0)
        """
        self.discriminator.train()
        batch_size = fake_ids.size(0)
        
        # Concatenate real and fake
        all_ids = torch.cat([real_ids, fake_ids], dim=0)
        all_lengths = torch.cat([real_lengths, fake_lengths], dim=0)
        all_labels = torch.cat([
            torch.ones(batch_size, device=self.device),   # real = 1
            torch.zeros(batch_size, device=self.device),  # fake = 0
        ], dim=0)
        
        # Forward
        self.d_optimizer.zero_grad()
        preds = self.discriminator(all_ids, all_lengths).squeeze(-1)  # (2*batch,)
        
        d_loss = self.bce_loss(preds, all_labels)
        d_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.get_trainable_params(), max_norm=1.0
        )
        self.d_optimizer.step()
        
        # Metrics
        with torch.no_grad():
            d_real = preds[:batch_size].mean().item()    # avg D(real)
            d_fake = preds[batch_size:].mean().item()    # avg D(fake)
        
        return {
            'd_loss': d_loss.item(),
            'd_reward_real': d_real,
            'd_reward_fake': d_fake,
        }

    # ────────────────────────────────────────────────────────────
    # STEP 3.3: Attacker (Policy Gradient) Update
    # ────────────────────────────────────────────────────────────

    def update_attacker(
        self,
        prompts: List[str],
        fake_texts: List[str],
        fake_ids: torch.Tensor,
        fake_lengths: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Step 3.3: Update Attacker via REINFORCE Policy Gradient.
        
        1. Get reward R from D(fake_text) [higher = better spoofing]
        2. Compute PPL penalty
        3. R_total = λ1 * R - λ2 * PPL_normalized
        4. ∇θ J(θ) = E[R_total · ∇θ log G_θ(y_t | y_{<t})]
        5. Update LoRA weights via gradient ascent
        """
        self.attacker.model.train()
        self.discriminator.eval()
        
        # ── 1) Reward from D ──
        reward_d = self.discriminator.get_reward(fake_ids, fake_lengths)  # (batch,)
        
        # ── 2) PPL penalty ──
        # Re-tokenize for the attacker model
        inputs = self.attacker.tokenizer(
            fake_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.adv_max_gen_length,
        ).to(self.device)
        
        with torch.no_grad():
            log_probs = self.attacker.compute_log_probs(
                inputs['input_ids'],
                inputs.get('attention_mask'),
            )
        
        # Per-sequence PPL
        seq_lengths = inputs['attention_mask'].sum(dim=-1).float() - 1  # -1 for shift
        seq_lengths = seq_lengths.clamp(min=1)
        avg_neg_log_prob = -(log_probs * inputs['attention_mask'][:, 1:]).sum(dim=-1) / seq_lengths
        ppls = torch.exp(avg_neg_log_prob)
        
        # Normalize PPL to [0, 1] range (softly)
        ppl_penalty = torch.sigmoid((ppls - 50) / 20)  # Center around PPL=50
        
        # ── 3) Total reward ──
        R_total = (
            self.config.adv_lambda_reward * reward_d 
            - self.config.adv_lambda_ppl * ppl_penalty
        )
        
        # ── 4) REINFORCE loss ──
        # Compute prompt lengths
        prompt_lengths = []
        for prompt in prompts:
            prompt_ids = self.attacker.tokenizer(prompt, add_special_tokens=True)['input_ids']
            prompt_lengths.append(len(prompt_ids))
        prompt_lengths_tensor = torch.tensor(prompt_lengths, device=self.device)
        
        self.g_optimizer.zero_grad()
        
        g_loss = self.attacker.reinforce_loss(
            generated_ids=inputs['input_ids'],
            rewards=R_total.detach(),
            prompt_lengths=prompt_lengths_tensor,
            baseline=self.config.adv_reward_baseline,
        )
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.attacker.get_lora_params(), max_norm=1.0)
        self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'avg_reward': R_total.mean().item(),
            'avg_ppl': ppls.mean().item(),
            'avg_d_score': reward_d.mean().item(),
        }

    # ────────────────────────────────────────────────────────────
    # EVALUATION (PHẦN 4)
    # ────────────────────────────────────────────────────────────

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate the GAN system (PHẦN 4).
        
        Metrics:
            - D_Loss: Should be stable (~0.5-0.7), not near 0
            - Spoofing Success Rate: % of fake texts D classifies as real
            - Diversity Score: Unique tokens ratio (mode collapse detection)
            - PPL: Quality of generated text
        """
        self.discriminator.eval()
        self.attacker.model.eval()
        
        # Generate evaluation samples
        num_eval = min(100, len(self.prompts))
        eval_prompts = self.prompts[:num_eval]
        
        all_fake_texts = []
        all_spoofing_results = []
        all_ppls = []
        
        batch_size = self.config.adv_batch_size
        
        for i in range(0, num_eval, batch_size):
            batch = eval_prompts[i:i+batch_size]
            
            # Generate fake
            fake_texts, fake_full_ids = self.attacker.generate(
                batch,
                max_length=self.config.adv_max_gen_length,
                temperature=self.config.adv_temperature,
                static_spoofer=self.static_spoofer,
            )
            all_fake_texts.extend(fake_texts)
            
            # Tokenize for D
            fake_token_ids_list = []
            for text in fake_texts:
                ids = self.attacker.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=self.config.adv_max_gen_length,
                    add_special_tokens=False,
                )["input_ids"][0]
                fake_token_ids_list.append(ids)
            
            pad_id = self.attacker.tokenizer.pad_token_id or 0
            fake_padded, fake_lengths = pad_sequences(fake_token_ids_list, pad_value=pad_id)
            fake_padded = fake_padded.to(self.device)
            fake_lengths = fake_lengths.to(self.device)
            
            # D score
            with torch.no_grad():
                d_scores = self.discriminator(fake_padded, fake_lengths).squeeze(-1)
                spoofed = (d_scores > 0.5).float()
                all_spoofing_results.extend(spoofed.cpu().tolist())
        
        # Compute metrics
        spoofing_rate = sum(all_spoofing_results) / max(len(all_spoofing_results), 1)
        
        # Diversity score
        all_token_ids = []
        for text in all_fake_texts:
            ids = self.attacker.tokenizer(text, add_special_tokens=False)['input_ids']
            all_token_ids.append(ids)
        diversity = compute_diversity_score(all_token_ids)
        
        metrics = {
            'spoofing_rate': spoofing_rate,
            'diversity': diversity,
        }
        
        # Check for mode collapse
        if diversity < 0.3:
            print(f"[EVAL] ⚠️  WARNING: Low diversity ({diversity:.3f}) — possible mode collapse!")
        
        return metrics

    # ────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        """
        Main adversarial training loop (PHẦN 3).
        
        For each epoch:
            3.1: Generation — G generates fake, Oracle generates real
            3.2: D update — train D on (real=1, fake=0)
            3.3: G update — REINFORCE with R from updated D
        
        Returns:
            Training history dict
        """
        print("=" * 60)
        print("STEP 3: ADVERSARIAL TRAINING LOOP")
        print("=" * 60)
        
        for epoch in range(self.config.adv_num_epochs):
            prompts = self._get_batch_prompts(epoch)
            
            # ── 3.1: Generation Phase ──
            result = self.generation_phase(prompts)
            fake_texts, real_texts, fake_ids, real_ids, fake_lengths, real_lengths = result
            
            # ── 3.2: Update Discriminator ──
            d_metrics = {}
            for _ in range(self.config.adv_d_steps):
                d_metrics = self.update_discriminator(
                    fake_ids, fake_lengths,
                    real_ids, real_lengths,
                )
            
            # ── 3.3: Update Attacker (Policy Gradient) ──
            g_metrics = {}
            for _ in range(self.config.adv_g_steps):
                g_metrics = self.update_attacker(
                    prompts, fake_texts,
                    fake_ids, fake_lengths,
                )
            
            # Record history
            all_metrics = {**d_metrics, **g_metrics}
            for key, value in all_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            # Log
            log_metrics(all_metrics, epoch)
            
            # ── Evaluation ──
            if (epoch + 1) % self.config.adv_eval_every == 0:
                eval_metrics = self.evaluate(epoch)
                print(f"[EVAL Epoch {epoch:04d}] " + 
                      " | ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items()))
                
                for key, value in eval_metrics.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)
            
            # ── Checkpoint ──
            if (epoch + 1) % self.config.adv_checkpoint_every == 0:
                self.save_checkpoint(epoch)
            
            # Memory cleanup
            del fake_texts, real_texts, fake_ids, real_ids, fake_lengths, real_lengths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("=" * 60)
        print("ADVERSARIAL TRAINING COMPLETE")
        print("=" * 60)
        
        return self.history

    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        ensure_dir(self.config.adv_checkpoint_dir)
        
        checkpoint = {
            'epoch': epoch,
            'discriminator_state': self.discriminator.state_dict(),
            'lora_state': {
                name: module.state_dict() 
                for name, module in self.attacker.lora_modules.items()
            },
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'history': self.history,
        }
        
        path = os.path.join(self.config.adv_checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        
        for name, state in checkpoint['lora_state'].items():
            if name in self.attacker.lora_modules:
                self.attacker.lora_modules[name].load_state_dict(state)
        
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.history = checkpoint['history']
        
        epoch = checkpoint['epoch']
        print(f"[Checkpoint] Loaded from {path} (epoch {epoch})")
        return epoch
