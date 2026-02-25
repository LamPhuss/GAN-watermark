# ============================================================
# main.py
# Description: Main entry point for the Watermark GAN
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

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


from utils.helpers import load_config, set_seed, ensure_dir


def run_pretrain_attacker(config):
    """Step 1: Pre-train Attacker via SFT on static spoofer data."""
    from training.pretrain_attacker import pretrain_attacker
    attacker = pretrain_attacker(config)
    return attacker


def run_pretrain_detector(config):
    """Step 2: Pre-train Discriminator on real vs natural text."""
    from training.pretrain_detector import pretrain_detector
    discriminator = pretrain_detector(config)
    return discriminator


def run_adversarial(config, attacker=None, discriminator=None):
    """Step 3: Adversarial training loop."""
    from training.adversarial_loop import AdversarialTrainer
    from data.data_generator import UPVOracle
    from models.detector import WatermarkDiscriminator
    from models.attacker import AttackerLLM
    
    device = config.device
    
    # Load or create models
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
        # Load pre-trained LoRA weights if available
        lora_path = os.path.join(config.adv_checkpoint_dir, "attacker_pretrained_lora.pt")
        if os.path.exists(lora_path):
            print(f"[Main] Loading LoRA weights from {lora_path}")
            lora_state = torch.load(lora_path, map_location=device)
            for name, state in lora_state.items():
                if name in attacker.lora_modules:
                    attacker.lora_modules[name].load_state_dict(state)
        else:
            print("[Main] Warning: No pre-trained LoRA weights found. Starting from scratch.")
    
    if discriminator is None:
        print("[Main] Loading pre-trained Discriminator...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.llm_name)
        vocab_size = tokenizer.vocab_size
        
        discriminator = WatermarkDiscriminator(
            vocab_size=vocab_size,
            embedding_dim=config.disc_embedding_dim,
            hidden_dim=config.disc_hidden_dim,
            num_lstm_layers=config.disc_num_lstm_layers,
            dropout=config.disc_dropout,
            freeze_embedding=config.disc_freeze_embedding,
        ).to(device)
        
        disc_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrained_best.pt")
        if os.path.exists(disc_path):
            print(f"[Main] Loading Discriminator weights from {disc_path}")
            discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        else:
            print("[Main] Warning: No pre-trained Discriminator found. Starting from scratch.")
    
    # Load Oracle (frozen UPV Generator)
    print("[Main] Loading UPV Oracle (frozen)...")
    oracle = UPVOracle(
        model_name=config.llm_name,
        device=device,
        upv_config_path=config.upv_config_path,
    )
    
    # Build static spoofer if available
    static_spoofer = getattr(attacker, '_static_spoofer', None)
    
    # Create trainer and run
    trainer = AdversarialTrainer(
        config=config,
        discriminator=discriminator,
        attacker=attacker,
        oracle=oracle,
        static_spoofer=static_spoofer,
    )
    
    # Check for existing checkpoint
    latest_checkpoint = None
    if os.path.isdir(config.adv_checkpoint_dir):
        checkpoints = [f for f in os.listdir(config.adv_checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoints:
            latest_checkpoint = os.path.join(
                config.adv_checkpoint_dir, 
                sorted(checkpoints)[-1]
            )
            print(f"[Main] Found checkpoint: {latest_checkpoint}")
    
    if latest_checkpoint:
        start_epoch = trainer.load_checkpoint(latest_checkpoint)
        print(f"[Main] Resuming from epoch {start_epoch}")
    
    # Run training
    history = trainer.train()
    
    return trainer, history


def run_evaluate(config):
    """Step 4: Full evaluation."""
    from evaluation.metrics import full_evaluation
    from data.data_generator import UPVOracle
    from models.detector import WatermarkDiscriminator
    from models.attacker import AttackerLLM
    import json
    
    device = config.device
    
    # Load models
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
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name)
    
    discriminator = WatermarkDiscriminator(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.disc_embedding_dim,
        hidden_dim=config.disc_hidden_dim,
        num_lstm_layers=config.disc_num_lstm_layers,
        dropout=config.disc_dropout,
        freeze_embedding=True,
    ).to(device)
    
    disc_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrained_best.pt")
    if os.path.exists(disc_path):
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))
    
    oracle = UPVOracle(
        model_name=config.llm_name,
        device=device,
        upv_config_path=config.upv_config_path,
    )
    
    # Load test prompts
    prompts = []
    try:
        with open(config.dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'prompt' in item:
                    prompts.append(item['prompt'])
    except:
        prompts = ["The latest research shows that"] * 50
    
    prompts = prompts[:100]  # Use 100 prompts for eval
    
    results = full_evaluation(
        discriminator=discriminator,
        attacker=attacker,
        oracle=oracle,
        prompts=prompts,
        config=config,
    )
    
    # Save results
    results_path = os.path.join(config.adv_checkpoint_dir, "evaluation_results.json")
    ensure_dir(config.adv_checkpoint_dir)
    
    # Convert numpy types to Python types
    serializable = {k: float(v) for k, v in results.items()}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"[Main] Evaluation results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Watermark GAN Training")
    parser.add_argument(
        '--stage', 
        type=str, 
        required=True,
        choices=['pretrain_attacker', 'pretrain_detector', 'adversarial', 'evaluate', 'all'],
        help='Training stage to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/gan_config.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    set_seed(args.seed)
    ensure_dir(config.adv_checkpoint_dir)
    
    print("=" * 60)
    print(f"WATERMARK GAN — Stage: {args.stage}")
    print(f"LLM: {config.llm_name}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    if args.stage == 'pretrain_attacker' or args.stage == 'all':
        attacker = run_pretrain_attacker(config)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        attacker = None
    
    if args.stage == 'pretrain_detector' or args.stage == 'all':
        discriminator = run_pretrain_detector(config)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        discriminator = None
    
    if args.stage == 'adversarial' or args.stage == 'all':
        trainer, history = run_adversarial(config, attacker, discriminator)
    
    if args.stage == 'evaluate' or args.stage == 'all':
        results = run_evaluate(config)
    
    print("\n[Main] Done!")


if __name__ == "__main__":
    main()
