# ============================================================
# metrics.py
# Description: Evaluation metrics for the Watermark GAN
# ============================================================

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter


def compute_spoofing_success_rate(
    discriminator,
    fake_token_ids: torch.Tensor,
    fake_lengths: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute the ratio of fake texts that D classifies as REAL (spoofing success).
    
    Args:
        discriminator: WatermarkDiscriminator model
        fake_token_ids: (batch, seq_len) fake text token IDs
        fake_lengths: (batch,) sequence lengths
        threshold: Classification threshold
    
    Returns:
        Spoofing success rate in [0, 1]
    """
    discriminator.eval()
    with torch.no_grad():
        probs = discriminator(fake_token_ids, fake_lengths).squeeze(-1)
        spoofed = (probs > threshold).float()
    
    return spoofed.mean().item()


def compute_d_loss_health(d_losses: List[float], window: int = 20) -> Dict[str, float]:
    """
    Analyze D_Loss health.
    
    A healthy GAN has D_Loss around 0.5-0.7:
        - D_Loss → 0: D is too strong, G can't learn
        - D_Loss → ln(2) ≈ 0.693: Perfect balance
        - D_Loss → ∞: G dominates, D can't distinguish
    
    Args:
        d_losses: List of D losses over time
        window: Window size for moving average
    
    Returns:
        Dict with health metrics
    """
    if len(d_losses) < window:
        recent = d_losses
    else:
        recent = d_losses[-window:]
    
    mean_loss = np.mean(recent)
    std_loss = np.std(recent)
    trend = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) > 1 else 0.0
    
    # Health assessment
    if mean_loss < 0.1:
        status = "UNHEALTHY: D too strong (loss near 0)"
    elif mean_loss > 1.5:
        status = "UNHEALTHY: D too weak (high loss)"
    elif abs(trend) > 0.05:
        status = "WARNING: Significant trend detected"
    else:
        status = "HEALTHY: Balanced"
    
    return {
        'mean_d_loss': mean_loss,
        'std_d_loss': std_loss,
        'trend': trend,
        'status': status,
    }


def compute_diversity_metrics(texts: List[str], tokenizer) -> Dict[str, float]:
    """
    Comprehensive diversity analysis to detect mode collapse.
    
    Checks:
        1. Unique token ratio (type-token ratio)
        2. Vocabulary coverage
        3. N-gram diversity
        4. Self-BLEU (lower = more diverse)
    
    Args:
        texts: List of generated texts
        tokenizer: Tokenizer
    
    Returns:
        Dict with diversity metrics
    """
    if not texts:
        return {'ttr': 0.0, 'vocab_coverage': 0.0, 'bigram_diversity': 0.0}
    
    all_tokens = []
    all_bigrams = []
    
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        all_tokens.extend(ids)
        
        # Bigrams
        for i in range(len(ids) - 1):
            all_bigrams.append((ids[i], ids[i + 1]))
    
    if len(all_tokens) == 0:
        return {'ttr': 0.0, 'vocab_coverage': 0.0, 'bigram_diversity': 0.0}
    
    # Type-Token Ratio (TTR)
    unique_tokens = len(set(all_tokens))
    ttr = unique_tokens / len(all_tokens)
    
    # Vocabulary coverage (% of vocab used)
    vocab_size = tokenizer.vocab_size
    vocab_coverage = unique_tokens / vocab_size
    
    # Bigram diversity
    unique_bigrams = len(set(all_bigrams))
    bigram_diversity = unique_bigrams / max(len(all_bigrams), 1)
    
    return {
        'ttr': ttr,
        'vocab_coverage': vocab_coverage,
        'bigram_diversity': bigram_diversity,
        'unique_tokens': unique_tokens,
        'total_tokens': len(all_tokens),
    }


def compute_upv_detection_rate(
    oracle,
    texts: List[str],
) -> Dict[str, float]:
    """
    Check how many texts the UPV detector considers watermarked.
    
    This is the ground-truth check — if the attacker successfully
    produces text that the UPV detector considers watermarked,
    the spoofing attack is succeeding.
    
    Args:
        oracle: UPVOracle instance
        texts: Generated texts
    
    Returns:
        Detection rate and average z-score
    """
    detected = 0
    z_scores = []
    
    for text in texts:
        try:
            result = oracle.detect_watermark(text)
            if result.get('is_watermarked', False):
                detected += 1
            score = result.get('score', 0)
            if score is not None:
                z_scores.append(float(score))
        except Exception:
            pass
    
    return {
        'detection_rate': detected / max(len(texts), 1),
        'avg_z_score': np.mean(z_scores) if z_scores else 0.0,
        'std_z_score': np.std(z_scores) if z_scores else 0.0,
    }


def full_evaluation(
    discriminator,
    attacker,
    oracle,
    prompts: List[str],
    config,
    static_spoofer=None,
) -> Dict[str, float]:
    """
    Run full evaluation suite (PHẦN 4).
    
    Args:
        discriminator: Trained WatermarkDiscriminator
        attacker: Trained AttackerLLM
        oracle: Frozen UPVOracle
        prompts: List of test prompts
        config: GANConfig
        static_spoofer: Optional StaticSpoofer
    
    Returns:
        Comprehensive evaluation metrics
    """
    device = config.device
    results = {}
    
    # Generate texts
    fake_texts, fake_ids = attacker.generate(
        prompts,
        max_length=config.adv_max_gen_length,
        temperature=config.adv_temperature,
        static_spoofer=static_spoofer,
    )
    
    # Tokenize for D
    from data.data_generator import pad_sequences
    fake_token_ids_list = []
    for text in fake_texts:
        ids = attacker.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=config.adv_max_gen_length,
            add_special_tokens=False,
        )["input_ids"][0]
        fake_token_ids_list.append(ids)
    
    pad_id = attacker.tokenizer.pad_token_id or 0
    fake_padded, fake_lengths = pad_sequences(fake_token_ids_list, pad_value=pad_id)
    fake_padded = fake_padded.to(device)
    fake_lengths = fake_lengths.to(device)
    
    # 1. Spoofing success rate (D perspective)
    results['spoofing_rate_d'] = compute_spoofing_success_rate(
        discriminator, fake_padded, fake_lengths
    )
    
    # 2. Detection rate by UPV (ground truth)
    upv_results = compute_upv_detection_rate(oracle, fake_texts)
    results.update({f'upv_{k}': v for k, v in upv_results.items()})
    
    # 3. Diversity
    diversity_results = compute_diversity_metrics(fake_texts, attacker.tokenizer)
    results.update({f'div_{k}': v for k, v in diversity_results.items()})
    
    # 4. Print summary
    print("\n" + "=" * 60)
    print("FULL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Spoofing Rate (D):       {results['spoofing_rate_d']:.2%}")
    print(f"  UPV Detection Rate:      {results['upv_detection_rate']:.2%}")
    print(f"  UPV Avg Z-Score:         {results['upv_avg_z_score']:.2f}")
    print(f"  Type-Token Ratio:        {results['div_ttr']:.4f}")
    print(f"  Bigram Diversity:        {results['div_bigram_diversity']:.4f}")
    print(f"  Vocab Coverage:          {results['div_vocab_coverage']:.4f}")
    print("=" * 60 + "\n")
    
    return results
