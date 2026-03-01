#!/usr/bin/env python3
"""
test_kgw.py — Verify KGW watermark works correctly before full migration.

Run from repo root:
    python scripts/test_kgw.py

Expected output:
  Oracle watermarked:  z-score > 4.0, green fraction > 60%
  Natural text:        z-score < 2.0, green fraction ~ 25%
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark.kgw_watermark import KGWOracle

print("=" * 60)
print("TEST: KGW-SelfHash Watermark")
print("=" * 60)

oracle = KGWOracle(
    model_name="facebook/opt-1.3b",
    device="cuda",
    gamma=0.25,
    delta=2.0,
    context_width=4,
    hash_key=15485863,
    z_threshold=4.0,
)

# Test 1: Watermarked text
print("\n--- Watermarked Text ---")
for i in range(5):
    wm_texts, _ = oracle.generate_watermarked(["The latest research in AI suggests that"])
    result = oracle.detect_watermark(wm_texts[0])
    print(f"[WM {i}] z={result['z_score']:6.2f} | green={result['green_fraction']:.2%} | "
          f"detected={result['is_watermarked']} | scored={result['num_scored']}")

# Test 2: Natural text
print("\n--- Natural (Unwatermarked) Text ---")
for i in range(5):
    nat_texts, _ = oracle.generate_unwatermarked(["The latest research in AI suggests that"])
    result = oracle.detect_watermark(nat_texts[0])
    print(f"[NAT {i}] z={result['z_score']:6.2f} | green={result['green_fraction']:.2%} | "
          f"detected={result['is_watermarked']} | scored={result['num_scored']}")

# Test 3: Spoofer stealing test (quick)
print("\n--- Spoofer Feasibility Test ---")
print("Collecting frequency statistics from 100 WM texts...")

from models.attacker import WatermarkLearner
learner = WatermarkLearner(tokenizer=oracle.tokenizer, prevctx_width=3)

wm_all, base_all = [], []
for i in range(10):
    wm_t, _ = oracle.generate_watermarked([
        "The economic impact", "Scientists discovered",
        "New technology enables", "The government announced",
        "Recent studies show", "Climate change affects",
        "The stock market", "Researchers at MIT",
        "Breaking news about", "The future of AI",
    ])
    nat_t, _ = oracle.generate_unwatermarked([
        "The economic impact", "Scientists discovered",
        "New technology enables", "The government announced",
        "Recent studies show", "Climate change affects",
        "The stock market", "Researchers at MIT",
        "Breaking news about", "The future of AI",
    ])
    wm_all.extend(wm_t)
    base_all.extend(nat_t)
    if (i + 1) % 5 == 0:
        print(f"  {(i+1)*10}/100 texts collected")

learner.learn_from_watermarked(wm_all)
learner.learn_from_baseline(base_all)

print(f"WM counts: {learner.counts_wm.total_counts():,}")
print(f"Base counts: {learner.counts_base.total_counts():,}")

# Build spoofer and test
from models.attacker import AttackerLLM
attacker = AttackerLLM(model_name="facebook/opt-1.3b", device="cuda")
vocab_size = attacker.model.config.vocab_size
spoofer = learner.build_spoofer(vocab_size, spoofer_strength=2.0)

print("\n--- Spoofer Output Test ---")
for i in range(5):
    texts, _ = attacker.generate(
        ["The latest research in AI suggests that"],
        max_length=256,
        temperature=1.0,
        static_spoofer=spoofer,
    )
    result = oracle.detect_watermark(texts[0])
    print(f"[SPF {i}] z={result['z_score']:6.2f} | green={result['green_fraction']:.2%} | "
          f"detected={result['is_watermarked']}")

print("\n" + "=" * 60)
print("If Spoofer z-scores are > 2.0 and green > 40%, migration is viable!")
print("=" * 60)
