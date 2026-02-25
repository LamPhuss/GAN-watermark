# ============================================================
# mc_search.py
# Description: Monte Carlo Search for chunk-level reward estimation.
#
# Adapted from SeqGAN (Yu et al., AAAI 2017) for the watermark
# adversarial training setting.
#
# Key idea:
#   Instead of giving Attacker a single reward for the entire
#   sequence (credit assignment problem), we divide the sequence
#   into K chunks and estimate per-chunk rewards via MC rollout.
#
#   For chunk k (not the last):
#     1. Attacker has generated prefix up to chunk boundary
#     2. Rollout policy completes the sequence N times
#     3. Discriminator scores each complete sequence
#     4. Q(chunk_k) = mean of N scores
#
#   For the last chunk:
#     Sequence is complete → Discriminator scores directly.
#
# Rollout policy:
#   Static Spoofer (from watermark-stealing) is used as rollout
#   policy to ensure suffix contains fake watermark signal,
#   avoiding the "clean suffix" problem.
# ============================================================

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MCSearchResult:
    """Result of MC Search generation."""
    generated_ids: torch.LongTensor   # (batch, seq_len) full generated sequences
    chunk_rewards: torch.Tensor       # (batch, num_chunks) per-chunk rewards
    prompt_lengths: torch.LongTensor  # (batch,) prompt lengths
    gen_lengths: torch.LongTensor     # (batch,) generated part lengths
    sequence_reward: torch.Tensor     # (batch,) final sequence-level reward


class MonteCarloSearch:
    """
    Monte Carlo Search with chunk-level reward estimation.

    Divides generated sequence into K chunks and estimates
    the value of each chunk via N rollouts using a rollout policy.

    Memory-efficient: all generation/rollout is done under torch.no_grad().
    Gradients only flow through reinforce_loss_mc() computation.
    """

    def __init__(
        self,
        num_chunks: int = 4,
        num_rollouts: int = 3,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        self.num_chunks = num_chunks
        self.num_rollouts = num_rollouts
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def generate_with_rewards(
        self,
        attacker,
        discriminator,
        prompts: List[str],
        max_new_tokens: int = 200,
        static_spoofer=None,
    ) -> MCSearchResult:
        """
        Generate text token-by-token with per-chunk MC rewards.

        Process:
          1. Tokenize prompts
          2. Generate max_new_tokens in K chunks
          3. After each chunk (except last): rollout N times → score → avg
          4. After last chunk: score directly

        Args:
            attacker:       AttackerLLM model
            discriminator:  UPVDiscriminatorWrapper (or any model with get_reward)
            prompts:        List of prompt strings
            max_new_tokens: Number of tokens to generate
            static_spoofer: StaticSpoofer for rollout policy (recommended)

        Returns:
            MCSearchResult with per-chunk rewards
        """
        attacker.model.eval()

        # Tokenize prompts
        inputs = attacker.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs["input_ids"]       # (batch, prompt_len)
        batch_size = input_ids.size(0)
        prompt_lengths = inputs["attention_mask"].sum(dim=-1)  # (batch,)

        # Calculate chunk boundaries
        chunk_size = max_new_tokens // self.num_chunks
        chunk_boundaries = []
        for k in range(self.num_chunks):
            end = (k + 1) * chunk_size
            if k == self.num_chunks - 1:
                end = max_new_tokens  # Last chunk gets remainder
            chunk_boundaries.append(end)

        # Generate token by token, collecting chunk rewards
        current_ids = input_ids.clone()
        chunk_rewards = torch.zeros(batch_size, self.num_chunks, device=self.device)
        tokens_generated = 0

        for chunk_k in range(self.num_chunks):
            chunk_end = chunk_boundaries[chunk_k]
            tokens_in_chunk = chunk_end - tokens_generated

            # ── Generate tokens for this chunk ──
            for step in range(tokens_in_chunk):
                logits = self._get_next_logits(attacker, current_ids)

                # Apply spoofer boost if available
                if static_spoofer is not None:
                    logits = self._apply_spoofer_boost(
                        logits, current_ids, static_spoofer, batch_size
                    )

                # Sample next token
                probs = F.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                # Early stop if all EOS
                if attacker.tokenizer.eos_token_id is not None:
                    if (next_token == attacker.tokenizer.eos_token_id).all():
                        break

            tokens_generated = chunk_end

            # ── Compute chunk reward ──
            if chunk_k < self.num_chunks - 1:
                # Not last chunk → MC rollout
                remaining_tokens = max_new_tokens - tokens_generated
                reward = self._mc_rollout_reward(
                    attacker=attacker,
                    discriminator=discriminator,
                    prefix_ids=current_ids,
                    remaining_tokens=remaining_tokens,
                    static_spoofer=static_spoofer,
                    batch_size=batch_size,
                )
            else:
                # Last chunk → direct evaluation
                reward = self._direct_reward(
                    discriminator=discriminator,
                    token_ids=current_ids,
                    prompt_lengths=prompt_lengths,
                )

            chunk_rewards[:, chunk_k] = reward

        # Compute final sequence reward (same as last chunk reward)
        sequence_reward = chunk_rewards[:, -1].clone()

        # Compute generation lengths
        gen_lengths = torch.full(
            (batch_size,), current_ids.size(1), device=self.device, dtype=torch.long
        ) - prompt_lengths

        return MCSearchResult(
            generated_ids=current_ids,
            chunk_rewards=chunk_rewards,
            prompt_lengths=prompt_lengths,
            gen_lengths=gen_lengths,
            sequence_reward=sequence_reward,
        )

    # ── Internal methods ──

    def _get_next_logits(
        self, attacker, input_ids: torch.LongTensor
    ) -> torch.Tensor:
        """Get logits for next token prediction."""
        outputs = attacker.model(input_ids)
        return outputs.logits[:, -1, :]  # (batch, vocab)

    def _apply_spoofer_boost(
        self,
        logits: torch.Tensor,
        input_ids: torch.LongTensor,
        spoofer,
        batch_size: int,
    ) -> torch.Tensor:
        """Apply static spoofer boost to logits."""
        for b in range(batch_size):
            ctx_width = spoofer.prevctx_width
            if input_ids.size(1) >= ctx_width:
                ctx = tuple(input_ids[b, -ctx_width:].cpu().tolist())
                boosts = spoofer.get_boosts(ctx, device=str(self.device))
                logits[b] += spoofer.spoofer_strength * boosts[: logits.size(-1)]
        return logits

    def _mc_rollout_reward(
        self,
        attacker,
        discriminator,
        prefix_ids: torch.LongTensor,
        remaining_tokens: int,
        static_spoofer,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Perform N rollouts from current prefix, score each, return mean.

        For each rollout:
          1. Copy prefix
          2. Use rollout policy (static spoofer) to complete the sequence
          3. Discriminator scores the complete sequence
          4. Average over N rollouts

        Returns:
            (batch,) mean reward across rollouts
        """
        if remaining_tokens <= 0:
            return self._direct_reward(
                discriminator, prefix_ids,
                prompt_lengths=None,
            )

        all_rewards = []

        for n in range(self.num_rollouts):
            # Copy prefix for this rollout
            rollout_ids = prefix_ids.clone()  # (batch, current_len)

            # Complete sequence using rollout policy
            for step in range(remaining_tokens):
                logits = self._get_next_logits(attacker, rollout_ids)

                # Rollout uses spoofer to ensure watermark-like suffix
                if static_spoofer is not None:
                    logits = self._apply_spoofer_boost(
                        logits, rollout_ids, static_spoofer, batch_size
                    )

                probs = F.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                rollout_ids = torch.cat([rollout_ids, next_token], dim=-1)

                # Early stop
                if attacker.tokenizer.eos_token_id is not None:
                    if (next_token == attacker.tokenizer.eos_token_id).all():
                        break

            # Score complete sequence
            reward = discriminator.get_reward(rollout_ids)  # (batch,)
            all_rewards.append(reward)

        # Mean over N rollouts: (batch,)
        stacked = torch.stack(all_rewards, dim=0)  # (N, batch)
        return stacked.mean(dim=0)

    def _direct_reward(
        self,
        discriminator,
        token_ids: torch.LongTensor,
        prompt_lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Score a complete sequence directly."""
        # Extract only generated part if prompt_lengths provided
        reward = discriminator.get_reward(token_ids, prompt_lengths)
        return reward  # (batch,)


def reinforce_loss_mc(
    attacker,
    generated_ids: torch.LongTensor,
    chunk_rewards: torch.Tensor,
    prompt_lengths: torch.LongTensor,
    num_chunks: int,
    baseline: float = 0.5,
    lambda_reward: float = 1.0,
    ppl_penalty: Optional[torch.Tensor] = None,
    lambda_ppl: float = 0.1,
) -> torch.Tensor:
    """
    Compute REINFORCE loss with per-chunk MC rewards.

    Instead of applying a single sequence reward to all tokens,
    each token receives the reward of its chunk:

        Loss = -E[ (R_chunk(t) - baseline) * log P(y_t | y_{<t}) ]

    This provides more precise credit assignment:
    - Early tokens get reward based on how well the prefix enables
      good completions (estimated via MC rollout)
    - Late tokens get reward based on the actual sequence quality

    Args:
        attacker:       AttackerLLM (for computing log probs)
        generated_ids:  (batch, seq_len) full sequences
        chunk_rewards:  (batch, num_chunks) per-chunk rewards from MC Search
        prompt_lengths: (batch,) prompt lengths
        num_chunks:     Number of chunks
        baseline:       Scalar baseline for variance reduction
        lambda_reward:  Weight for discriminator reward
        ppl_penalty:    (batch,) optional PPL penalty per sequence
        lambda_ppl:     Weight for PPL penalty

    Returns:
        Scalar loss (negate of expected reward — minimize this)
    """
    # Compute log probs (THIS is where gradients flow)
    log_probs = attacker.compute_log_probs(generated_ids)  # (batch, seq_len-1)

    batch_size = generated_ids.size(0)
    seq_len = log_probs.size(1)

    # Create prompt mask: only count generated tokens
    prompt_mask = torch.zeros(batch_size, seq_len, device=generated_ids.device)
    for b in range(batch_size):
        start = max(0, prompt_lengths[b].item() - 1)
        prompt_mask[b, start:] = 1.0

    # Count generated tokens per sequence
    gen_token_counts = prompt_mask.sum(dim=-1).clamp(min=1)  # (batch,)

    # Map each token position to its chunk reward
    # Token t (in generated part) belongs to chunk k = t * num_chunks // gen_length
    token_rewards = torch.zeros_like(log_probs)  # (batch, seq_len-1)

    for b in range(batch_size):
        gen_start = max(0, prompt_lengths[b].item() - 1)
        gen_len = seq_len - gen_start

        if gen_len <= 0:
            continue

        for t in range(gen_start, seq_len):
            # Which chunk does this generated token belong to?
            gen_pos = t - gen_start
            chunk_k = min(
                int(gen_pos * num_chunks / gen_len),
                num_chunks - 1,
            )
            token_rewards[b, t] = chunk_rewards[b, chunk_k]

    # Apply PPL penalty if provided
    if ppl_penalty is not None:
        total_reward = lambda_reward * token_rewards
        # Broadcast PPL penalty (per-sequence) to all tokens
        for b in range(batch_size):
            gen_start = max(0, prompt_lengths[b].item() - 1)
            total_reward[b, gen_start:] -= lambda_ppl * ppl_penalty[b]
    else:
        total_reward = lambda_reward * token_rewards

    # Advantage = reward - baseline
    advantage = total_reward - baseline  # (batch, seq_len-1)

    # REINFORCE: Loss = -E[advantage * log_prob]
    # Mask out prompt tokens
    weighted_log_probs = advantage.detach() * log_probs * prompt_mask
    loss = -weighted_log_probs.sum() / gen_token_counts.sum()

    return loss
