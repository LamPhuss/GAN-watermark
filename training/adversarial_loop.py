# ============================================================
# adversarial_loop.py — KGW version
# D = z-score (non-trainable), G = Attacker with LoRA
# ============================================================

import os, gc, json, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from models.attacker import AttackerLLM, StaticSpoofer
from watermark.kgw_watermark import KGWOracle
from watermark.kgw_discriminator import KGWDiscriminator
from training.mc_search import MonteCarloSearch, MCSearchResult, reinforce_loss_mc
from data.data_generator import pad_sequences
from utils.helpers import (
    GANConfig, log_metrics, ensure_dir,
    compute_ppl_from_logprobs, compute_diversity_score,
)


class AdversarialTrainer:
    """
    Adversarial Training with KGW z-score discriminator.

    Key difference from UPV version:
    - D is deterministic z-score computation → no D training needed
    - Reward = sigmoid((z_score - z_center) / temperature)
    - Only G is updated via REINFORCE
    """

    def __init__(
        self,
        config: GANConfig,
        discriminator: KGWDiscriminator,
        attacker: AttackerLLM,
        oracle: KGWOracle,
        static_spoofer: Optional[StaticSpoofer] = None,
    ):
        self.config = config
        self.device = config.device
        self.discriminator = discriminator
        self.attacker = attacker
        self.oracle = oracle
        self.static_spoofer = static_spoofer

        # MC Search
        self.mc_search = MonteCarloSearch(
            num_chunks=config.mc_num_chunks,
            num_rollouts=config.mc_num_rollouts,
            temperature=config.adv_temperature,
            device=config.device,
        )

        # Only G optimizer needed (D is not trained)
        self.g_optimizer = torch.optim.Adam(
            self.attacker.get_lora_params(),
            lr=config.adv_g_lr,
        )

        self.prompts = self._load_prompts()
        self.history: Dict[str, List[float]] = {}

        print(f"[AdversarialTrainer] KGW Discriminator (z-score, non-trainable)")
        print(f"[AdversarialTrainer] MC Search: {config.mc_num_chunks} chunks, "
              f"{config.mc_num_rollouts} rollouts")
        print(f"[AdversarialTrainer] {len(self.prompts)} prompts loaded")

    def _load_prompts(self) -> List[str]:
        prompts = []
        try:
            with open(self.config.dataset_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'prompt' in item:
                        prompts.append(item['prompt'])
        except Exception:
            prompts = ["The latest research shows that"] * 100
        return prompts[:self.config.num_prompts]

    def _get_batch_prompts(self, epoch: int) -> List[str]:
        start = (epoch * self.config.mc_batch_size) % len(self.prompts)
        batch = []
        for i in range(self.config.mc_batch_size):
            batch.append(self.prompts[(start + i) % len(self.prompts)])
        return batch

    # ── Generation Phase ──

    def generation_phase_mc(self, prompts: List[str]):
        """Generate with MC Search, get z-score rewards from KGW discriminator."""
        self.attacker.model.eval()

        mc_result = self.mc_search.search(
            model=self.attacker,
            prompts=prompts,
            max_length=self.config.adv_max_gen_length,
            reward_fn=self._compute_reward,
            static_spoofer=self.static_spoofer,
        )

        # Generate real watermarked text for comparison metrics
        real_texts, _ = self.oracle.generate_watermarked(prompts)
        pad_id = self.attacker.tokenizer.pad_token_id or 0
        real_ids_list = []
        for text in real_texts:
            ids = self.attacker.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=self.config.adv_max_gen_length,
                add_special_tokens=False,
            )["input_ids"][0]
            real_ids_list.append(ids)
        real_padded, real_lengths = pad_sequences(real_ids_list, pad_value=pad_id)
        real_padded = real_padded.to(self.device)
        real_lengths = real_lengths.to(self.device)

        return mc_result, real_padded, real_lengths

    def _compute_reward(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Compute z-score reward for generated sequences."""
        texts = self.attacker.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        rewards = []
        for text in texts:
            reward = self.discriminator.get_reward(text, self.attacker.tokenizer)
            rewards.append(reward)
        return torch.tensor(rewards, device=self.device)

    # ── G Update ──

    def update_attacker(self, mc_result: MCSearchResult) -> Dict[str, float]:
        self.attacker.model.train()
        self.g_optimizer.zero_grad()

        g_loss = reinforce_loss_mc(
            model=self.attacker,
            mc_result=mc_result,
            lambda_reward=self.config.adv_lambda_reward,
            lambda_ppl=self.config.adv_lambda_ppl,
            baseline=self.config.adv_reward_baseline,
        )

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.attacker.get_lora_params(), max_norm=1.0)
        self.g_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'chunk_rewards_mean': mc_result.chunk_rewards.mean().item(),
        }

    # ── Evaluation ──

    def evaluate(self, epoch: int) -> Dict[str, float]:
        self.attacker.model.eval()
        num_eval = min(50, len(self.prompts))
        eval_prompts = self.prompts[:num_eval]
        batch_size = self.config.mc_batch_size

        all_z_scores = []
        all_token_ids = []

        for i in range(0, num_eval, batch_size):
            batch = eval_prompts[i:i+batch_size]
            fake_texts, _ = self.attacker.generate(
                batch,
                max_length=self.config.adv_max_gen_length,
                temperature=self.config.adv_temperature,
                static_spoofer=self.static_spoofer,
            )
            for text in fake_texts:
                result = self.oracle.detect_watermark(text)
                all_z_scores.append(result['z_score'])
                ids = self.attacker.tokenizer(text, add_special_tokens=False)["input_ids"]
                all_token_ids.append(ids)

        avg_z = sum(all_z_scores) / max(len(all_z_scores), 1)
        spoofing_rate = sum(1 for z in all_z_scores if z > self.config.wm_z_threshold) / max(len(all_z_scores), 1)
        diversity = compute_diversity_score(all_token_ids)

        metrics = {
            'avg_z_score': avg_z,
            'spoofing_rate': spoofing_rate,
            'diversity': diversity,
        }

        if diversity < 0.3:
            print(f"[EVAL] ⚠️  Low diversity ({diversity:.3f}) — possible mode collapse!")

        return metrics

    # ── Main Loop ──

    def train(self) -> Dict[str, List[float]]:
        print("=" * 60)
        print("ADVERSARIAL TRAINING (KGW z-score)")
        print(f"  D: z-score (non-trainable)")
        print(f"  Chunks: {self.config.mc_num_chunks}")
        print(f"  Rollouts: {self.config.mc_num_rollouts}")
        print("=" * 60)

        for epoch in range(self.config.adv_num_epochs):
            prompts = self._get_batch_prompts(epoch)

            # ── MC Generation ──
            mc_result, real_padded, real_lengths = self.generation_phase_mc(prompts)

            # ── No D update needed (z-score is deterministic) ──
            d_metrics = {
                'd_loss': 0.0,
                'd_reward_real': 0.0,
                'd_reward_fake': mc_result.chunk_rewards.mean().item(),
            }

            # ── G update ──
            g_metrics = {}
            for _ in range(self.config.adv_g_steps):
                g_metrics = self.update_attacker(mc_result)

            all_metrics = {**d_metrics, **g_metrics}
            for key, value in all_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

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

            del mc_result, real_padded, real_lengths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("=" * 60)
        print("ADVERSARIAL TRAINING COMPLETE")
        print("=" * 60)
        return self.history

    def save_checkpoint(self, epoch: int):
        ensure_dir(self.config.adv_checkpoint_dir)
        checkpoint = {
            'epoch': epoch,
            'lora_state': {
                name: module.state_dict()
                for name, module in self.attacker.lora_modules.items()
            },
            'g_optimizer': self.g_optimizer.state_dict(),
            'history': self.history,
        }
        path = os.path.join(self.config.adv_checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        for name, state in checkpoint['lora_state'].items():
            if name in self.attacker.lora_modules:
                self.attacker.lora_modules[name].load_state_dict(state)
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.history = checkpoint['history']
        epoch = checkpoint['epoch']
        print(f"[Checkpoint] Loaded from {path} (epoch {epoch})")
        return epoch
