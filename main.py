# ============================================================
# main.py
# Description: Main entry point for Watermark GAN
#   with Monte Carlo Search + UPV Detector
#
# Usage:
#   python main.py --stage pretrain_attacker
#   python main.py --stage pretrain_detector
#   python main.py --stage adversarial
#   python main.py --stage evaluate
#   python main.py --stage all
# ============================================================

import os
import sys
import argparse
import gc
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.helpers import load_config, set_seed, ensure_dir


def run_pretrain_attacker(config):
    """Step 1: Pre-train Attacker via SFT on static spoofer data."""
    from training.pretrain_attacker import pretrain_attacker
    return pretrain_attacker(config)


def run_pretrain_detector(config):
    """Step 2: Pre-train UPV Detector on real vs natural text."""
    from training.pretrain_detector import pretrain_detector
    return pretrain_detector(config)


def run_adversarial(config, attacker=None, discriminator=None):
    """Step 3: Adversarial training with MC Search."""
    from training.adversarial_loop import AdversarialTrainer
    from data.data_generator import UPVOracle
    from models.upv_discriminator import UPVDiscriminatorWrapper
    from models.attacker import AttackerLLM

    device = config.device

    # ── Load Attacker ──
    if attacker is None:
        print("[Main] Loading pre-trained Attacker...")
        attacker = AttackerLLM(
            model_name=config.llm_name,
            device=device,
            lora_r=config.att_lora_r,
            lora_alpha=config.att_lora_alpha,
            lora_dropout=config.att_lora_dropout,
            lora_target_modules=config.att_lora_target_modules,
        )
        lora_path = os.path.join(config.adv_checkpoint_dir, "attacker_pretrained_lora.pt")
        if os.path.exists(lora_path):
            print(f"[Main] Loading LoRA weights from {lora_path}")
            lora_state = torch.load(lora_path, map_location=device)
            for name, state in lora_state.items():
                if name in attacker.lora_modules:
                    attacker.lora_modules[name].load_state_dict(state)
        else:
            print("[Main] Warning: No pre-trained LoRA weights found.")

    # ── Load UPV Discriminator ──
    if discriminator is None:
        print("[Main] Loading UPV Detector as Discriminator...")
        discriminator = UPVDiscriminatorWrapper(
            bit_number=config.disc_bit_number,
            detector_weights_path=config.upv_detector_weights,
            freeze_embedding=config.disc_freeze_embedding,
            device=device,
        ).to(device)

        disc_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrained_best.pt")
        if os.path.exists(disc_path):
            print(f"[Main] Loading pre-trained D weights from {disc_path}")
            discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        else:
            print("[Main] Warning: No pre-trained D found. Using original UPV weights.")

    # ── Load Oracle ──
    print("[Main] Loading UPV Oracle (frozen)...")
    oracle = UPVOracle(
        model_name=config.llm_name,
        device=device,
        upv_config_path=config.upv_config_path,
    )

    # ── Build static spoofer ──
    static_spoofer = getattr(attacker, '_static_spoofer', None)
    if static_spoofer is None:
        print("[Main] Warning: No static spoofer available. MC rollout will use raw LM.")

    # ── Create trainer ──
    trainer = AdversarialTrainer(
        config=config,
        discriminator=discriminator,
        attacker=attacker,
        oracle=oracle,
        static_spoofer=static_spoofer,
    )

    # Resume from checkpoint if available
    latest_checkpoint = None
    if os.path.isdir(config.adv_checkpoint_dir):
        checkpoints = [
            f for f in os.listdir(config.adv_checkpoint_dir)
            if f.startswith('checkpoint_epoch_')
        ]
        if checkpoints:
            latest_checkpoint = os.path.join(
                config.adv_checkpoint_dir, sorted(checkpoints)[-1]
            )

    if latest_checkpoint:
        start_epoch = trainer.load_checkpoint(latest_checkpoint)
        print(f"[Main] Resuming from epoch {start_epoch}")

    history = trainer.train()
    return trainer, history


def run_evaluate(config):
    """Step 4: Full evaluation."""
    from evaluation.metrics import full_evaluation
    from data.data_generator import UPVOracle
    from models.upv_discriminator import UPVDiscriminatorWrapper
    from models.attacker import AttackerLLM
    import json

    device = config.device

    # Load Attacker
    attacker = AttackerLLM(
        model_name=config.llm_name,
        device=device,
        lora_r=config.att_lora_r,
        lora_alpha=config.att_lora_alpha,
        lora_dropout=config.att_lora_dropout,
        lora_target_modules=config.att_lora_target_modules,
    )
    lora_path = os.path.join(config.adv_checkpoint_dir, "attacker_pretrained_lora.pt")
    if os.path.exists(lora_path):
        lora_state = torch.load(lora_path, map_location=device)
        for name, state in lora_state.items():
            if name in attacker.lora_modules:
                attacker.lora_modules[name].load_state_dict(state)

    # Load UPV Discriminator
    discriminator = UPVDiscriminatorWrapper(
        bit_number=config.disc_bit_number,
        detector_weights_path=config.upv_detector_weights,
        freeze_embedding=True,
        device=device,
    ).to(device)

    disc_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrained_best.pt")
    if os.path.exists(disc_path):
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))

    # Load Oracle
    oracle = UPVOracle(
        model_name=config.llm_name,
        device=device,
        upv_config_path=config.upv_config_path,
    )

    # Load prompts
    prompts = []
    try:
        with open(config.dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'prompt' in item:
                    prompts.append(item['prompt'])
    except Exception:
        prompts = ["The latest research shows that"] * 50
    prompts = prompts[:100]

    results = full_evaluation(
        discriminator=discriminator,
        attacker=attacker,
        oracle=oracle,
        prompts=prompts,
        config=config,
    )

    results_path = os.path.join(config.adv_checkpoint_dir, "evaluation_results.json")
    ensure_dir(config.adv_checkpoint_dir)
    serializable = {k: float(v) for k, v in results.items()}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"[Main] Results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Watermark GAN (MC Search + UPV)")
    parser.add_argument(
        '--stage', type=str, required=True,
        choices=['pretrain_attacker', 'pretrain_detector', 'adversarial', 'evaluate', 'all'],
    )
    parser.add_argument('--config', type=str, default='config/gan_config.yaml')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    set_seed(args.seed)
    ensure_dir(config.adv_checkpoint_dir)

    print("=" * 60)
    print(f"WATERMARK GAN (MC Search + UPV Detector)")
    print(f"  Stage: {args.stage}")
    print(f"  LLM: {config.llm_name}")
    print(f"  MC Chunks: {config.mc_num_chunks}, Rollouts: {config.mc_num_rollouts}")
    print(f"  Device: {config.device}")
    print("=" * 60)

    attacker = None
    discriminator = None

    if args.stage in ('pretrain_attacker', 'all'):
        attacker = run_pretrain_attacker(config)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.stage in ('pretrain_detector', 'all'):
        discriminator = run_pretrain_detector(config)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.stage in ('adversarial', 'all'):
        run_adversarial(config, attacker, discriminator)

    if args.stage in ('evaluate', 'all'):
        run_evaluate(config)

    print("\n[Main] Done!")


if __name__ == "__main__":
    main()
