# ============================================================
# adversarial_loop.py
# Description: Step 3 — GAN Adversarial Training Loop
#   with Monte Carlo Search and UPV Detector as Discriminator.
#
# Changes from original:
#   - Uses UPVDiscriminatorWrapper (original UPV detector) as D
#   - MC Search provides per-chunk rewards for Attacker RL update
#   - Attacker update uses reinforce_loss_mc (chunk-level)
#   - Discriminator trained on real watermark (1) vs fake watermark (0)
# ============================================================

import os
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from models.upv_discriminator import UPVDiscriminatorWrapper
from models.attacker import AttackerLLM, StaticSpoofer
from data.data_generator import UPVOracle, pad_sequences
from training.mc_search import MonteCarloSearch, MCSearchResult, reinforce_loss_mc
from utils.helpers import (
    GANConfig, set_seed, log_metrics, ensure_dir,
    compute_ppl_from_logprobs, compute_diversity_score
)


class AdversarialTrainer:
    """
    GAN Adversarial Training with MC Search + UPV Detector.

    The Minimax Game:
        - D_φ (UPV Detector) maximizes: E[log D(real)] + E[log(1 - D(fake))]
        - G_θ (Attacker+LoRA) maximizes: E[D(fake)] via MC-REINFORCE

    Each epoch:
        1. MC Search: G generates fake text with per-chunk reward estimation
        2. Oracle generates real watermarked text
        3. D is updated on (real=1, fake=0)
        4. G is updated via REINFORCE with chunk-level rewards from MC Search
    """

    def __init__(
        self,
        config: GANConfig,
        discriminator: UPVDiscriminatorWrapper,
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

        # Monte Carlo Search
        self.mc_search = MonteCarloSearch(
            num_chunks=config.mc_num_chunks,
            num_rollouts=config.mc_num_rollouts,
            temperature=config.adv_temperature,
            device=config.device,
        )

        # Optimizers
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.get_trainable_params(),
            lr=config.adv_d_lr,
        )
        self.g_optimizer = torch.optim.Adam(
            self.attacker.get_lora_params(),
            lr=config.adv_g_lr,
        )

        # Loss for Discriminator
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
            'chunk_rewards_mean': [],
        }

        # Log MC Search config
        print(f"[AdversarialTrainer] MC Search: {config.mc_num_chunks} chunks, "
              f"{config.mc_num_rollouts} rollouts")
        print(f"[AdversarialTrainer] Discriminator: UPV Detector (original)")
        params = self.discriminator.count_params()
        print(f"[AdversarialTrainer] D params — total: {params['total']:,}, "
              f"trainable: {params['trainable']:,}, frozen: {params['frozen']:,}")
        print(f"[AdversarialTrainer] {len(self.prompts)} prompts loaded")

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
        except Exception:
            prompts = [
                "The latest research in artificial intelligence suggests that",
                "In a groundbreaking study, scientists discovered that",
                "The economic impact of climate change has been",
            ] * 100
        return prompts

    def _get_batch_prompts(self, epoch: int) -> List[str]:
        """Get a batch of prompts for the current epoch."""
        batch_size = self.config.mc_batch_size
        start = (epoch * batch_size) % len(self.prompts)
        batch = []
        for i in range(batch_size):
            idx = (start + i) % len(self.prompts)
            batch.append(self.prompts[idx])
        return batch

    # ────────────────────────────────────────────────────────────
    # STEP 3.1: Generation Phase with MC Search
    # ────────────────────────────────────────────────────────────

    def generation_phase_mc(
        self, prompts: List[str]
    ) -> Tuple[MCSearchResult, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generation phase using Monte Carlo Search.

        1. Attacker generates fake text with per-chunk MC rewards
        2. Oracle generates real watermarked text (same prompts)

        Returns:
            (mc_result, real_padded, real_lengths, fake_padded_for_d)
        """
        # ── Fake: Attacker generates with MC Search ──
        mc_result = self.mc_search.generate_with_rewards(
            attacker=self.attacker,
            discriminator=self.discriminator,
            prompts=prompts,
            max_new_tokens=self.config.adv_max_gen_length,
            static_spoofer=self.static_spoofer,
        )

        # ── Real: Oracle generates ──
        real_texts, _ = self.oracle.generate_watermarked(
            prompts,
            max_new_tokens=self.config.adv_max_gen_length,
        )

        # Tokenize real texts using attacker's tokenizer
        real_ids_list = []
        for text in real_texts:
            ids = self.attacker.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=self.config.adv_max_gen_length,
                add_special_tokens=False,
            )["input_ids"][0]
            real_ids_list.append(ids)

        pad_id = self.attacker.tokenizer.pad_token_id or 0
        real_padded, real_lengths = pad_sequences(real_ids_list, pad_value=pad_id)
        real_padded = real_padded.to(self.device)
        real_lengths = real_lengths.to(self.device)

        return mc_result, real_padded, real_lengths

    # ────────────────────────────────────────────────────────────
    # STEP 3.2: Discriminator Update
    # ────────────────────────────────────────────────────────────

    def update_discriminator(
        self,
        fake_ids: torch.Tensor,
        real_ids: torch.Tensor,
        real_lengths: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update UPV Discriminator.

        D is trained to output:
            - 1 for real watermarked text (from UPV Generator)
            - 0 for fake/spoofed text (from Attacker)

        Loss = BCE(D(real), 1) + BCE(D(fake), 0)
        """
        self.discriminator.train()

        batch_real = real_ids.size(0)
        batch_fake = fake_ids.size(0)

        # Pad to same length
        max_len = max(real_ids.size(1), fake_ids.size(1))
        pad_id = 0

        if real_ids.size(1) < max_len:
            pad_real = torch.full(
                (batch_real, max_len - real_ids.size(1)),
                pad_id, dtype=torch.long, device=self.device,
            )
            real_ids_padded = torch.cat([real_ids, pad_real], dim=1)
        else:
            real_ids_padded = real_ids[:, :max_len]

        if fake_ids.size(1) < max_len:
            pad_fake = torch.full(
                (batch_fake, max_len - fake_ids.size(1)),
                pad_id, dtype=torch.long, device=self.device,
            )
            fake_ids_padded = torch.cat([fake_ids, pad_fake], dim=1)
        else:
            fake_ids_padded = fake_ids[:, :max_len]

        # Fake lengths (approximate from non-pad tokens)
        fake_lengths = (fake_ids_padded != pad_id).sum(dim=1).clamp(min=1)

        # Concatenate
        all_ids = torch.cat([real_ids_padded, fake_ids_padded], dim=0)
        all_lengths = torch.cat([real_lengths, fake_lengths], dim=0)
        all_labels = torch.cat([
            torch.ones(batch_real, device=self.device),
            torch.zeros(batch_fake, device=self.device),
        ], dim=0)

        # Forward + backward
        self.d_optimizer.zero_grad()
        preds = self.discriminator(all_ids, all_lengths).squeeze(-1)

        d_loss = self.bce_loss(preds, all_labels)
        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.discriminator.get_trainable_params(), max_norm=1.0
        )
        self.d_optimizer.step()

        # Metrics
        with torch.no_grad():
            d_real = preds[:batch_real].mean().item()
            d_fake = preds[batch_real:].mean().item()

        return {
            'd_loss': d_loss.item(),
            'd_reward_real': d_real,
            'd_reward_fake': d_fake,
        }

    # ────────────────────────────────────────────────────────────
    # STEP 3.3: Attacker Update (MC-REINFORCE)
    # ────────────────────────────────────────────────────────────

    def update_attacker(
        self,
        mc_result: MCSearchResult,
    ) -> Dict[str, float]:
        """
        Update Attacker via REINFORCE with per-chunk MC rewards.

        Uses chunk-level rewards from MC Search for better
        credit assignment than sequence-level REINFORCE.
        """
        self.attacker.model.train()
        self.discriminator.eval()

        generated_ids = mc_result.generated_ids
        chunk_rewards = mc_result.chunk_rewards
        prompt_lengths = mc_result.prompt_lengths

        # ── PPL penalty (optional) ──
        ppl_penalty = None
        if self.config.adv_lambda_ppl > 0:
            with torch.no_grad():
                log_probs = self.attacker.compute_log_probs(generated_ids)
                # Mean neg log prob per sequence
                gen_mask = torch.zeros_like(log_probs)
                for b in range(generated_ids.size(0)):
                    start = max(0, prompt_lengths[b].item() - 1)
                    gen_mask[b, start:] = 1.0
                gen_counts = gen_mask.sum(dim=-1).clamp(min=1)
                avg_nll = -(log_probs * gen_mask).sum(dim=-1) / gen_counts
                ppls = torch.exp(avg_nll)
                ppl_penalty = torch.sigmoid((ppls - 50) / 20)

        # ── MC-REINFORCE loss ──
        self.g_optimizer.zero_grad()

        g_loss = reinforce_loss_mc(
            attacker=self.attacker,
            generated_ids=generated_ids,
            chunk_rewards=chunk_rewards,
            prompt_lengths=prompt_lengths,
            num_chunks=self.config.mc_num_chunks,
            baseline=self.config.adv_reward_baseline,
            lambda_reward=self.config.adv_lambda_reward,
            ppl_penalty=ppl_penalty,
            lambda_ppl=self.config.adv_lambda_ppl,
        )

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.attacker.get_lora_params(), max_norm=1.0)
        self.g_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'avg_chunk_reward': chunk_rewards.mean().item(),
            'chunk_reward_std': chunk_rewards.std().item(),
            'avg_seq_reward': mc_result.sequence_reward.mean().item(),
        }

    # ────────────────────────────────────────────────────────────
    # EVALUATION
    # ────────────────────────────────────────────────────────────

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """Evaluate spoofing success and diversity."""
        self.discriminator.eval()
        self.attacker.model.eval()

        num_eval = min(50, len(self.prompts))
        eval_prompts = self.prompts[:num_eval]
        batch_size = self.config.mc_batch_size

        all_spoofing = []
        all_token_ids = []

        for i in range(0, num_eval, batch_size):
            batch = eval_prompts[i : i + batch_size]

            # Generate fake text (no MC search needed for eval — just generate)
            fake_texts, fake_full_ids = self.attacker.generate(
                batch,
                max_length=self.config.adv_max_gen_length,
                temperature=self.config.adv_temperature,
                static_spoofer=self.static_spoofer,
            )

            # Tokenize for D
            fake_ids_list = []
            for text in fake_texts:
                ids = self.attacker.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=self.config.adv_max_gen_length,
                    add_special_tokens=False,
                )["input_ids"][0]
                fake_ids_list.append(ids)

            pad_id = self.attacker.tokenizer.pad_token_id or 0
            fake_padded, fake_lengths = pad_sequences(fake_ids_list, pad_value=pad_id)
            fake_padded = fake_padded.to(self.device)
            fake_lengths = fake_lengths.to(self.device)

            with torch.no_grad():
                d_scores = self.discriminator(fake_padded, fake_lengths).squeeze(-1)
                spoofed = (d_scores > 0.5).float()
                all_spoofing.extend(spoofed.cpu().tolist())

            for text in fake_texts:
                ids = self.attacker.tokenizer(text, add_special_tokens=False)["input_ids"]
                all_token_ids.append(ids)

        spoofing_rate = sum(all_spoofing) / max(len(all_spoofing), 1)
        diversity = compute_diversity_score(all_token_ids)

        metrics = {
            'spoofing_rate': spoofing_rate,
            'diversity': diversity,
        }

        if diversity < 0.3:
            print(f"[EVAL] ⚠️  Low diversity ({diversity:.3f}) — possible mode collapse!")

        return metrics

    # ────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        """
        Main adversarial training loop with MC Search.

        For each epoch:
            3.1: MC Generation — G generates with chunk-level reward estimation
            3.2: D update — train UPV Detector on (real=1, fake=0)
            3.3: G update — MC-REINFORCE with per-chunk rewards
        """
        print("=" * 60)
        print("ADVERSARIAL TRAINING (MC Search + UPV Detector)")
        print(f"  Chunks: {self.config.mc_num_chunks}")
        print(f"  Rollouts per chunk: {self.config.mc_num_rollouts}")
        print(f"  Batch size: {self.config.mc_batch_size}")
        print(f"  Max gen length: {self.config.adv_max_gen_length}")
        print("=" * 60)

        for epoch in range(self.config.adv_num_epochs):
            prompts = self._get_batch_prompts(epoch)

            # ── 3.1: MC Generation Phase ──
            mc_result, real_padded, real_lengths = self.generation_phase_mc(prompts)

            # ── 3.2: Update Discriminator ──
            d_metrics = {}
            for _ in range(self.config.adv_d_steps):
                d_metrics = self.update_discriminator(
                    fake_ids=mc_result.generated_ids,
                    real_ids=real_padded,
                    real_lengths=real_lengths,
                )

            # ── 3.3: Update Attacker (MC-REINFORCE) ──
            g_metrics = {}
            for _ in range(self.config.adv_g_steps):
                g_metrics = self.update_attacker(mc_result)

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
                print(
                    f"[EVAL Epoch {epoch:04d}] "
                    + " | ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
                )
                for key, value in eval_metrics.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)

            # ── Checkpoint ──
            if (epoch + 1) % self.config.adv_checkpoint_every == 0:
                self.save_checkpoint(epoch)

            # Memory cleanup
            del mc_result, real_padded, real_lengths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("=" * 60)
        print("ADVERSARIAL TRAINING COMPLETE")
        print("=" * 60)
        return self.history

    # ── Checkpoint ──

    def save_checkpoint(self, epoch: int) -> None:
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
        path = os.path.join(
            self.config.adv_checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt"
        )
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, path: str) -> int:
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
