#!/usr/bin/env python3
"""
test_kgw_v2.py — Verify Spoofer feasibility with sufficient data.

Paper dùng 30,000 queries → millions of counts.
Test này dùng 1,000 texts → ~250k counts (đủ để thấy signal).

Run: python scripts/test_kgw_v2.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from watermark.kgw_watermark import KGWOracle

print("=" * 60)
print("TEST: KGW Spoofer Feasibility (1,000 texts)")
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

# ── Load prompts from large_dataset.jsonl ──
import json

DATASET_PATH = "data/large_dataset.jsonl"
PROMPTS = []
with open(DATASET_PATH, 'r') as f:
    for line in f:
        item = json.loads(line)
        if 'prompt' in item:
            PROMPTS.append(item['prompt'])

print(f"Loaded {len(PROMPTS)} prompts from {DATASET_PATH}")

NUM_TEXTS = min(10000, len(PROMPTS))
BATCH_SIZE = 10
NUM_BATCHES = NUM_TEXTS // BATCH_SIZE

# ── Cache paths ──
CACHE_DIR = "checkpoints"
os.makedirs(CACHE_DIR, exist_ok=True)
WM_CACHE = os.path.join(CACHE_DIR, "kgw_learning_wm_texts.json")
BASE_CACHE = os.path.join(CACHE_DIR, "kgw_learning_base_texts.json")

if os.path.exists(WM_CACHE) and os.path.exists(BASE_CACHE):
    print(f"\n📂 Loading cached data from {CACHE_DIR}/")
    with open(WM_CACHE, 'r') as f:
        wm_all = json.load(f)
    with open(BASE_CACHE, 'r') as f:
        base_all = json.load(f)
    print(f"  Loaded {len(wm_all)} WM + {len(base_all)} base texts from cache")
else:
    wm_all, base_all = [], []
    t0 = time.time()
    print(f"\nCollecting {NUM_TEXTS} WM + {NUM_TEXTS} base texts...")
    for i in range(NUM_BATCHES):
        batch = PROMPTS[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        wm_t, _ = oracle.generate_watermarked(batch)
        nat_t, _ = oracle.generate_unwatermarked(batch)
        wm_all.extend(wm_t)
        base_all.extend(nat_t)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {len(wm_all)}/{NUM_TEXTS} texts | {elapsed:.0f}s elapsed")

    print(f"\nTotal: {len(wm_all)} WM + {len(base_all)} base texts in {time.time()-t0:.0f}s")

    # ── Save to cache ──
    print(f"\n💾 Saving to {CACHE_DIR}/ for reuse...")
    with open(WM_CACHE, 'w') as f:
        json.dump(wm_all, f)
    with open(BASE_CACHE, 'w') as f:
        json.dump(base_all, f)
    print(f"  Saved: {WM_CACHE} ({os.path.getsize(WM_CACHE)/1024/1024:.1f} MB)")
    print(f"  Saved: {BASE_CACHE} ({os.path.getsize(BASE_CACHE)/1024/1024:.1f} MB)")

# ── Learn watermark patterns ──
print("\n--- Learning Phase ---")
from models.attacker import WatermarkLearner

learner = WatermarkLearner(tokenizer=oracle.tokenizer, prevctx_width=3)
learner.learn_from_watermarked(wm_all)
learner.learn_from_baseline(base_all)

print(f"WM counts:   {learner.counts_wm.total_counts():>10,}")
print(f"Base counts: {learner.counts_base.total_counts():>10,}")

# ── Build spoofer ──
from models.attacker import AttackerLLM

attacker = AttackerLLM(model_name="facebook/opt-1.3b", device="cuda")
vocab_size = attacker.model.config.vocab_size
spoofer = learner.build_spoofer(vocab_size, spoofer_strength=2.0)

# ── Test spoofer output ──
print("\n--- Spoofer Output (20 samples) ---")
spf_z_scores = []
test_prompts = PROMPTS[:20]  # use first 20 unique prompts
for i, prompt in enumerate(test_prompts):
    texts, _ = attacker.generate(
        [prompt],
        max_length=256,
        temperature=1.0,
        static_spoofer=spoofer,
    )
    result = oracle.detect_watermark(texts[0])
    z = result['z_score']
    gf = result['green_fraction']
    spf_z_scores.append(z)
    print(f"[SPF {i:2d}] z={z:6.2f} | green={gf:.2%} | detected={result['is_watermarked']}")

avg_z = sum(spf_z_scores) / len(spf_z_scores)
detected_count = sum(1 for z in spf_z_scores if z > 4.0)

print(f"\n{'=' * 60}")
print(f"RESULTS:")
print(f"  Avg Spoofer z-score: {avg_z:.2f}")
print(f"  Detected as WM:     {detected_count}/20 ({detected_count/20:.0%})")
print(f"  WM counts collected: {learner.counts_wm.total_counts():,}")

if avg_z > 2.0:
    print(f"\n  ✅ Spoofer WORKS! Migration to KGW is viable.")
elif avg_z > 0.5:
    print(f"\n  ⚠️  Spoofer shows WEAK signal. May need more data or tuning.")
    print(f"     Try increasing to 5,000-10,000 texts in full pipeline.")
else:
    print(f"\n  ❌ Spoofer NOT working. Check prevctx_width alignment.")
print(f"{'=' * 60}")