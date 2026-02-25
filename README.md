# GAN-watermark
# Watermark GAN: UPV Generator + Attacker Adversarial Training

## Architecture Overview

This project implements a Multi-agent Minimax Game for watermark spoofing:

1. **UPV Generator (Oracle)**: Frozen network from MarkLLM that generates real watermarked text
2. **UPV Detector (Discriminator D_φ)**: LSTM-based network that distinguishes real vs fake watermarks
3. **Attacker (Generator G_θ)**: LLM with LoRA that learns to produce spoofed watermarked text

## Directory Structure

```
watermark_gan/
├── config/                  # Configuration files
│   └── gan_config.yaml
├── models/                  # Model definitions
│   ├── detector.py          # LSTM Discriminator (D_φ)
│   └── attacker.py          # LLM Attacker with LoRA (G_θ)
├── training/                # Training loops
│   ├── pretrain_detector.py # Step 2: Pre-train D
│   ├── pretrain_attacker.py # Step 1: Pre-train G via SFT
│   └── adversarial_loop.py  # Step 3: GAN adversarial training
├── data/                    # Data generation utilities
│   └── data_generator.py    # Generate real/fake watermark data
├── evaluation/              # Evaluation metrics
│   └── metrics.py           # D_Loss, Spoofing Rate, Diversity
├── utils/                   # Utilities
│   └── helpers.py
├── main.py                  # Main entry point
└── README.md
```

## How to Run

```bash
# Step 1: Pre-train Attacker (Static Spoofer -> SFT)
python main.py --stage pretrain_attacker

# Step 2: Pre-train Detector 
python main.py --stage pretrain_detector

# Step 3: Adversarial Training Loop
python main.py --stage adversarial

# Evaluate
python main.py --stage evaluate
```
