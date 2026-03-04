#!/usr/bin/env python3
# ============================================================
# main.py — Watermark GAN (KGW-SelfHash)
# ============================================================

import os, sys, argparse, gc, json
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.helpers import load_config, set_seed, ensure_dir


def run_pretrain_attacker(config):
    from training.pretrain_attacker import pretrain_attacker
    return pretrain_attacker(config)


def run_adversarial(config, attacker=None, discriminator=None):
    from training.adversarial_loop import AdversarialTrainer
    from watermark.kgw_watermark import KGWOracle
    from watermark.kgw_discriminator import KGWDiscriminator
    from models.attacker import AttackerLLM, WatermarkLearner

    device = config.device

    # ── Attacker ──
    if attacker is None:
        attacker = AttackerLLM(
            model_name=config.llm_name, device=device,
            lora_r=config.att_lora_r, lora_alpha=config.att_lora_alpha,
            lora_dropout=config.att_lora_dropout,
            lora_target_modules=config.att_lora_target_modules,
        )
        lora_path = os.path.join(config.adv_checkpoint_dir, "attacker_pretrained_lora.pt")
        if os.path.exists(lora_path):
            lora_state = torch.load(lora_path, map_location=device)
            for name, state in lora_state.items():
                if name in attacker.lora_modules:
                    attacker.lora_modules[name].load_state_dict(state)

    # ── KGW Discriminator (non-trainable) ──
    if discriminator is None:
        discriminator = KGWDiscriminator(
            vocab_size=attacker.model.config.vocab_size,
            gamma=config.wm_gamma, delta=config.wm_delta,
            context_width=config.wm_context_width, hash_key=config.wm_hash_key,
            z_center=config.disc_z_center, temperature=config.disc_temperature,
            device=device,
        )

    # ── Oracle ──
    oracle = KGWOracle(
        model_name=config.llm_name, device=device,
        gamma=config.wm_gamma, delta=config.wm_delta,
        context_width=config.wm_context_width, hash_key=config.wm_hash_key,
    )

    # ── Spoofer ──
    static_spoofer = getattr(attacker, '_static_spoofer', None)
    if static_spoofer is None:
        wm_path = os.path.join(config.adv_checkpoint_dir, "kgw_learning_wm_texts.json")
        base_path = os.path.join(config.adv_checkpoint_dir, "kgw_learning_base_texts.json")
        if os.path.exists(wm_path) and os.path.exists(base_path):
            with open(wm_path) as f:
                wm_texts = json.load(f)
            with open(base_path) as f:
                base_texts = json.load(f)
            learner = WatermarkLearner(tokenizer=attacker.tokenizer, prevctx_width=config.att_prevctx_width)
            learner.learn_from_watermarked(wm_texts)
            learner.learn_from_baseline(base_texts)
            static_spoofer = learner.build_spoofer(
                attacker.model.config.vocab_size,
                spoofer_strength=config.att_spoofer_strength,
            )
            attacker._static_spoofer = static_spoofer
            print(f"[Main] Spoofer rebuilt. WM: {learner.counts_wm.total_counts():,}")

    trainer = AdversarialTrainer(
        config=config, discriminator=discriminator,
        attacker=attacker, oracle=oracle, static_spoofer=static_spoofer,
    )

    # Resume
    if os.path.isdir(config.adv_checkpoint_dir):
        cks = [f for f in os.listdir(config.adv_checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if cks:
            path = os.path.join(config.adv_checkpoint_dir, sorted(cks)[-1])
            start = trainer.load_checkpoint(path)
            print(f"[Main] Resuming from epoch {start}")

    return trainer.train()


def run_evaluate(config):
    from watermark.kgw_watermark import KGWOracle
    from models.attacker import AttackerLLM
    import numpy as np

    device = config.device
    attacker = AttackerLLM(
        model_name=config.llm_name, device=device,
        lora_r=config.att_lora_r, lora_alpha=config.att_lora_alpha,
        lora_dropout=config.att_lora_dropout,
        lora_target_modules=config.att_lora_target_modules,
    )
    lora_path = os.path.join(config.adv_checkpoint_dir, "attacker_pretrained_lora.pt")
    if os.path.exists(lora_path):
        lora_state = torch.load(lora_path, map_location=device)
        for name, state in lora_state.items():
            if name in attacker.lora_modules:
                attacker.lora_modules[name].load_state_dict(state)

    oracle = KGWOracle(
        model_name=config.llm_name, device=device,
        gamma=config.wm_gamma, delta=config.wm_delta,
        context_width=config.wm_context_width, hash_key=config.wm_hash_key,
    )

    prompts = []
    try:
        with open(config.dataset_path) as f:
            for line in f:
                item = json.loads(line)
                if 'prompt' in item:
                    prompts.append(item['prompt'])
    except:
        prompts = ["The latest research shows that"] * 50
    prompts = prompts[:100]

    z_scores, detected = [], 0
    for i, prompt in enumerate(prompts):
        texts, _ = attacker.generate([prompt], max_length=200, temperature=1.0)
        result = oracle.detect_watermark(texts[0])
        z_scores.append(result['z_score'])
        if result['is_watermarked']:
            detected += 1
        if (i + 1) % 20 == 0:
            print(f"[Eval] {i+1}/{len(prompts)} | avg z={np.mean(z_scores):.2f}")

    results = {
        'avg_z_score': float(np.mean(z_scores)),
        'std_z_score': float(np.std(z_scores)),
        'spoofing_rate': detected / len(prompts),
        'num_samples': len(prompts),
    }
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    results_path = os.path.join(config.adv_checkpoint_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Watermark GAN (KGW-SelfHash)")
    parser.add_argument('--stage', type=str, required=True,
                       choices=['pretrain_attacker', 'adversarial', 'evaluate', 'all'])
    parser.add_argument('--config', type=str, default='config/gan_config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = load_config(os.path.join(project_root, args.config))
    set_seed(args.seed)
    ensure_dir(config.adv_checkpoint_dir)

    print("=" * 60)
    print(f"WATERMARK GAN (KGW-SelfHash)")
    print(f"  Stage: {args.stage}")
    print(f"  KGW: γ={config.wm_gamma}, δ={config.wm_delta}, h={config.wm_context_width-1}")
    print("=" * 60)

    attacker = None
    if args.stage in ('pretrain_attacker', 'all'):
        attacker = run_pretrain_attacker(config)
        gc.collect(); torch.cuda.empty_cache()

    if args.stage in ('adversarial', 'all'):
        run_adversarial(config, attacker)

    if args.stage in ('evaluate', 'all'):
        run_evaluate(config)

    print("\n[Main] Done!")


if __name__ == "__main__":
    main()
