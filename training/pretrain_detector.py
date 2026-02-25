# ============================================================
# pretrain_detector.py
# Description: Step 2 — Pre-train the UPV Detector (D_φ)
#   on real watermarked vs natural text.
#
# The UPV Detector (original from the paper) is loaded with
# pre-trained weights, then fine-tuned on:
#   - Real watermarked text → label 1
#   - Natural text → label 0
#
# binary_classifier (shared embedding) is UNFROZEN during
# pre-training for better adaptation, then FROZEN before
# adversarial training begins.
# ============================================================

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

from models.upv_discriminator import UPVDiscriminatorWrapper
from data.data_generator import UPVOracle, DataGenerator, pad_sequences
from utils.helpers import GANConfig, set_seed, log_metrics, ensure_dir


class BinaryTokenDataset(Dataset):
    """
    Dataset for UPV Detector pre-training.
    
    Stores pre-tokenized sequences as token IDs.
    The UPVDiscriminatorWrapper handles token→binary conversion.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.labels = labels
        self.token_ids = []
        for text in texts:
            ids = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_length, add_special_tokens=False,
            )["input_ids"][0]
            self.token_ids.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


def collate_fn(batch, pad_value=0):
    """Collate with padding."""
    token_ids_list, labels = zip(*batch)
    padded, lengths = pad_sequences(list(token_ids_list), pad_value=pad_value)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels


def pretrain_detector(config: GANConfig) -> UPVDiscriminatorWrapper:
    """
    Pre-train UPV Detector on real watermarked vs natural text.

    Steps:
        1. Load UPV Detector with original pre-trained weights
        2. Generate training data (watermarked + natural)
        3. Fine-tune LSTM + FC layers (unfreeze embedding during pretrain)
        4. Freeze embedding before returning

    Returns:
        UPVDiscriminatorWrapper ready for adversarial training
    """
    set_seed(42)
    device = config.device

    print("=" * 60)
    print("STEP 2: PRE-TRAINING UPV DETECTOR")
    print("=" * 60)

    # ── 1) Generate data or load cached ──
    data_cache_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrain_data.json")

    if os.path.exists(data_cache_path):
        print(f"[PretrainD] Loading cached data from {data_cache_path}")
        with open(data_cache_path, 'r') as f:
            data = json.load(f)
        real_texts = data['real_watermarked']
        natural_texts = data['natural']
    else:
        print("[PretrainD] Generating training data...")
        oracle = UPVOracle(
            model_name=config.llm_name,
            device=device,
            upv_config_path=config.upv_config_path,
        )
        data_gen = DataGenerator(oracle, config.dataset_path)
        real_texts, natural_texts = data_gen.generate_discriminator_data(
            num_samples=config.disc_pretrain_num_samples,
            save_path=data_cache_path,
        )

    # ── 2) Build datasets ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_texts = real_texts + natural_texts
    all_labels = [1] * len(real_texts) + [0] * len(natural_texts)

    import random
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)

    split_idx = int(0.8 * len(all_texts))
    train_dataset = BinaryTokenDataset(
        list(all_texts[:split_idx]), list(all_labels[:split_idx]),
        tokenizer, max_length=256,
    )
    val_dataset = BinaryTokenDataset(
        list(all_texts[split_idx:]), list(all_labels[split_idx:]),
        tokenizer, max_length=256,
    )

    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.disc_pretrain_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.disc_pretrain_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ── 3) Initialize UPV Detector ──
    discriminator = UPVDiscriminatorWrapper(
        bit_number=config.disc_bit_number,
        detector_weights_path=config.upv_detector_weights,
        freeze_embedding=False,  # UNFREEZE during pre-training
        device=device,
    ).to(device)

    params = discriminator.count_params()
    print(f"[PretrainD] UPV Detector — total: {params['total']:,}, "
          f"trainable: {params['trainable']:,}")
    print(f"[PretrainD] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        discriminator.detector.parameters(),  # All params (unfrozen for pretrain)
        lr=config.disc_pretrain_lr,
    )

    # ── 4) Training loop ──
    best_f1 = 0.0

    for epoch in range(config.disc_pretrain_epochs):
        # Train
        discriminator.train()
        train_loss = 0.0
        num_batches = 0

        for padded, lengths, labels in train_loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = discriminator(padded, lengths).squeeze(-1)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.detector.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        # Validate
        discriminator.eval()
        all_preds, all_labels_list = [], []
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for padded, lengths, labels in val_loader:
                padded = padded.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                preds = discriminator(padded, lengths).squeeze(-1)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_batches += 1

                all_preds.extend((preds > 0.5).float().cpu().tolist())
                all_labels_list.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / max(val_batches, 1)
        f1 = f1_score(all_labels_list, all_preds, zero_division=0)
        precision = precision_score(all_labels_list, all_preds, zero_division=0)
        recall = recall_score(all_labels_list, all_preds, zero_division=0)

        log_metrics({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'f1': f1, 'precision': precision, 'recall': recall,
        }, epoch)

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrained_best.pt")
            ensure_dir(config.adv_checkpoint_dir)
            torch.save(discriminator.state_dict(), save_path)
            print(f"[PretrainD] Best model (F1={f1:.4f}) saved to {save_path}")

        if f1 >= 0.95:
            print(f"[PretrainD] Target F1 reached at epoch {epoch}!")
            break

    # ── 5) FREEZE embedding before adversarial training ──
    discriminator.freeze_shared_embedding()
    print(f"[PretrainD] Done. Best F1: {best_f1:.4f}. Embedding FROZEN for adversarial phase.")

    return discriminator
