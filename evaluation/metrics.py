# ============================================================
# metrics.py — KGW version
# ============================================================

import numpy as np
from typing import Dict, List


def compute_kgw_detection_rate(oracle, texts: List[str]) -> Dict[str, float]:
    """Check how many texts the KGW detector considers watermarked."""
    detected = 0
    z_scores = []
    green_fractions = []

    for text in texts:
        try:
            result = oracle.detect_watermark(text)
            if result.get('is_watermarked', False):
                detected += 1
            z = result.get('z_score', 0)
            if z is not None:
                z_scores.append(float(z))
            gf = result.get('green_fraction', 0)
            if gf is not None:
                green_fractions.append(float(gf))
        except Exception:
            pass

    return {
        'detection_rate': detected / max(len(texts), 1),
        'avg_z_score': np.mean(z_scores) if z_scores else 0.0,
        'std_z_score': np.std(z_scores) if z_scores else 0.0,
        'avg_green_fraction': np.mean(green_fractions) if green_fractions else 0.0,
    }


def compute_text_diversity(texts: List[str], tokenizer) -> Dict[str, float]:
    if not texts:
        return {'ttr': 0.0, 'bigram_diversity': 0.0}

    all_tokens, all_bigrams = [], []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        all_tokens.extend(ids)
        for i in range(len(ids) - 1):
            all_bigrams.append((ids[i], ids[i + 1]))

    if not all_tokens:
        return {'ttr': 0.0, 'bigram_diversity': 0.0}

    return {
        'ttr': len(set(all_tokens)) / len(all_tokens),
        'bigram_diversity': len(set(all_bigrams)) / max(len(all_bigrams), 1),
        'unique_tokens': len(set(all_tokens)),
        'total_tokens': len(all_tokens),
    }


def full_evaluation(discriminator, attacker, oracle, prompts, config, static_spoofer=None):
    """Full evaluation suite."""
    fake_texts = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i:i+4]
        texts, _ = attacker.generate(
            batch, max_length=config.adv_max_gen_length,
            temperature=config.adv_temperature,
            static_spoofer=static_spoofer,
        )
        fake_texts.extend(texts)

    detection = compute_kgw_detection_rate(oracle, fake_texts)
    diversity = compute_text_diversity(fake_texts, attacker.tokenizer)

    results = {**detection, **diversity}
    print(f"\n[Evaluation] Spoofing rate: {detection['detection_rate']:.2%}")
    print(f"[Evaluation] Avg z-score: {detection['avg_z_score']:.2f}")
    print(f"[Evaluation] Diversity (TTR): {diversity['ttr']:.4f}")
    return results
