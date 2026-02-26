# ============================================================
# pretrain_detector.py
# Description: Step 2 — Pre-train the UPV Detector (D_φ)
#
# FIXES APPLIED:
#   FIX 2: binary_classifier (shared embedding) is FROZEN during
#          ALL phases (pretrain AND adversarial). NEVER unfreeze.
#          Optimizer only gets LSTM + FC params.
#   FIX 3: Support "random_tokens" data mode (matches original
#          unforgeable_watermark repo) alongside "llm_text" mode.
#   FIX 4: Default hyperparams now match original repo:
#          epochs=80, lr=0.0005, num_samples=10000
# ============================================================

import os
import sys
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

from models.upv_discriminator import UPVDiscriminatorWrapper
from data.data_generator import UPVOracle, DataGenerator, pad_sequences
from utils.helpers import GANConfig, set_seed, log_metrics, ensure_dir


# ════════════════════════════════════════════════════════════
# Dataset classes
# ════════════════════════════════════════════════════════════

class BinaryTokenDataset(Dataset):
    """
    Dataset for UPV Detector pre-training using LLM text.
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


class RandomTokenBinaryDataset(Dataset):
    """
    FIX 3: Dataset for UPV Detector pre-training using random token
    sequences + z-score labeling. Matches original unforgeable_watermark
    repo exactly.

    Each sample is:
      - A random token ID sequence of length ~200
      - Label = 1 if z_score > z_value, else 0

    Data is stored as pre-computed binary vectors (batch, seq_len, bit_number)
    so it bypasses _tokens_to_binary() and feeds directly into UPVDetector.
    """

    def __init__(self, data: List[Tuple[torch.Tensor, int]]):
        """
        Args:
            data: List of (binary_tensor, label) where
                  binary_tensor is (seq_len, bit_number) float tensor
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ════════════════════════════════════════════════════════════
# Collate functions
# ════════════════════════════════════════════════════════════

def collate_fn_token_ids(batch, pad_value=0):
    """Collate for BinaryTokenDataset (token IDs)."""
    token_ids_list, labels = zip(*batch)
    padded, lengths = pad_sequences(list(token_ids_list), pad_value=pad_value)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels


def collate_fn_binary(batch):
    """
    Collate for RandomTokenBinaryDataset (binary vectors).
    Pads binary tensors to same seq_len.
    """
    binary_list, labels = zip(*batch)
    max_len = max(b.size(0) for b in binary_list)
    bit_number = binary_list[0].size(1)

    padded = torch.zeros(len(binary_list), max_len, bit_number)
    lengths = torch.zeros(len(binary_list), dtype=torch.long)

    for i, b in enumerate(binary_list):
        seq_len = b.size(0)
        padded[i, :seq_len, :] = b
        lengths[i] = seq_len

    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels


# ════════════════════════════════════════════════════════════
# FIX 3: Random token sequence data generation
# ════════════════════════════════════════════════════════════

def _int_to_bin_list(n: int, length: int = 16) -> List[int]:
    """Convert integer to binary list (MSB first)."""
    return [int(b) for b in format(n, 'b').zfill(length)]


def generate_random_token_data(config: GANConfig) -> List[Tuple[torch.Tensor, int]]:
    """
    FIX 3: Generate detector training data using random token sequences
    + z-score labeling. Matches original unforgeable_watermark repo.

    Process (from original watermark_model.py):
      1. For each sample: pick random green_ratio ∈ [0, 1]
      2. Generate token list of length 200 with that green_ratio
      3. Compute z-score using watermark generator
      4. Label = 1 if z_score > z_threshold, else 0
      5. Convert token list to binary vectors

    Returns:
        List of (binary_tensor (seq_len, bit_number), label)
    """
    # Import architecture from train_upv (which has WatermarkEngine)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)

    from scripts.train_upv import (
        BinaryClassifier, SubNet, int_to_bin_list, get_value, max_number
    )

    bit_number = config.bit_number
    window_size = config.window_size
    layers = config.layers
    z_threshold = config.z_threshold
    gamma = config.gamma
    sigma = config.sigma
    num_samples = config.disc_pretrain_num_samples
    device = config.device

    # Load the trained Generator (BinaryClassifier)
    generator_path = config.upv_generator_weights
    # We need combine_model.pt (full BinaryClassifier), not sub_net.pt
    combine_path = generator_path.replace("generator_model", "combine_model")
    if not os.path.exists(combine_path):
        # Try to find it in the same directory
        model_dir = os.path.dirname(generator_path)
        combine_path = os.path.join(model_dir, "combine_model.pt")

    if not os.path.exists(combine_path):
        print(f"[PretrainD] WARNING: combine_model.pt not found at {combine_path}")
        print("[PretrainD] Falling back to llm_text mode.")
        return None

    model = BinaryClassifier(bit_number, window_size, layers)
    model.load_state_dict(torch.load(combine_path, map_location=device))
    model = model.to(device).eval()
    print(f"[PretrainD] Loaded BinaryClassifier from {combine_path}")

    vocab = list(range(1, 2 ** bit_number - 1))
    cache = {}
    from math import sqrt

    def _get_value(input_x):
        with torch.no_grad():
            return (model(input_x) > 0.5).bool().item()

    def _judge_green(input_ids, current_number):
        last_nums = input_ids[-(window_size - 1):] if window_size - 1 > 0 else []
        pair = list(last_nums) + [current_number]
        key = tuple(int(x) for x in pair)
        bin_list = [_int_to_bin_list(int(n), bit_number) for n in pair]
        if key in cache:
            return cache[key]
        result = _get_value(torch.FloatTensor(bin_list).unsqueeze(0).to(device))
        cache[key] = result
        return result

    def _random_sample(input_ids, is_green):
        last_nums = input_ids[-(window_size - 1):] if window_size - 1 > 0 else []
        while True:
            number = random.choice(vocab)
            pair = list(last_nums) + [number]
            key = tuple(int(x) for x in pair)
            bin_list = [_int_to_bin_list(int(n), bit_number) for n in pair]
            if key in cache:
                result = cache[key]
            else:
                result = _get_value(torch.FloatTensor(bin_list).unsqueeze(0).to(device))
                cache[key] = result
            if is_green and result:
                return number
            elif not is_green and not result:
                return number

    def _generate_list_with_green_ratio(length, green_ratio):
        token_list = random.sample(vocab, window_size - 1) if window_size - 1 > 0 else random.sample(vocab, 1)
        is_green = []
        while len(token_list) < length:
            green = 1 if random.random() < green_ratio else 0
            if green:
                token = _random_sample(torch.LongTensor(token_list), True)
                token_list.append(token)
                is_green.append(1)
            else:
                token = _random_sample(torch.LongTensor(token_list), False)
                token_list.append(token)
                is_green.append(0)

        # Wrap-around for initial tokens
        is_green_prepend = []
        for i in range(0, window_size - 1):
            if window_size - 1 > 0:
                tail_slice = token_list[-(window_size - 1 - i):]
                head_slice = token_list[:i]
                input_slice = tail_slice + head_slice
            else:
                input_slice = []
            is_green_prepend.append(
                1 if _judge_green(torch.LongTensor(input_slice), token_list[i]) else 0
            )
        is_green = is_green_prepend + is_green
        return token_list, is_green

    def _compute_z_score(green_count, T):
        numer = green_count - gamma * T
        denom = sqrt(T * gamma * (1 - gamma) + sigma * sigma * T)
        return numer / denom

    # Generate data
    data = []
    print(f"[PretrainD] Generating {num_samples} random token sequences...")

    for i in range(num_samples):
        length = 200
        green_ratio = random.random()
        token_list, is_green = _generate_list_with_green_ratio(length, green_ratio)

        green_count = sum(is_green)
        z_score = _compute_z_score(green_count, len(token_list))
        label = 1 if z_score > z_threshold else 0

        # Convert to binary vectors: (seq_len, bit_number)
        bin_tensor = torch.tensor(
            [_int_to_bin_list(t, bit_number) for t in token_list],
            dtype=torch.float32,
        )
        data.append((bin_tensor, label))

        if (i + 1) % 1000 == 0:
            pos = sum(1 for _, l in data if l == 1)
            print(f"  [{i+1}/{num_samples}] pos={pos}, neg={len(data)-pos}")

    pos_count = sum(1 for _, l in data if l == 1)
    print(f"[PretrainD] Generated {len(data)} samples: {pos_count} positive, {len(data)-pos_count} negative")
    return data


# ════════════════════════════════════════════════════════════
# Training loop (binary data mode — FIX 3)
# ════════════════════════════════════════════════════════════

def _train_on_binary_data(
    discriminator: UPVDiscriminatorWrapper,
    train_data: List[Tuple[torch.Tensor, int]],
    config: GANConfig,
    device: str,
) -> UPVDiscriminatorWrapper:
    """
    Train detector on pre-computed binary vectors (random token mode).
    Data bypasses _tokens_to_binary() and feeds directly into detector.
    """
    # Split train/val
    random.shuffle(train_data)
    split_idx = int(0.8 * len(train_data))
    train_dataset = RandomTokenBinaryDataset(train_data[:split_idx])
    val_dataset = RandomTokenBinaryDataset(train_data[split_idx:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.disc_pretrain_batch_size,
        shuffle=True,
        collate_fn=collate_fn_binary,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.disc_pretrain_batch_size,
        shuffle=False,
        collate_fn=collate_fn_binary,
    )

    params = discriminator.count_params()
    print(f"[PretrainD] UPV Detector — total: {params['total']:,}, "
          f"trainable: {params['trainable']:,}, frozen: {params['frozen']:,}")
    print(f"[PretrainD] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    criterion = nn.BCELoss()

    # FIX 2: Only trainable params (LSTM + FC, NOT binary_classifier)
    optimizer = torch.optim.Adam(
        discriminator.get_trainable_params(),
        lr=config.disc_pretrain_lr,
    )

    best_f1 = 0.0

    for epoch in range(config.disc_pretrain_epochs):
        # ── Train ──
        discriminator.train()
        train_loss = 0.0
        num_batches = 0

        for binary_padded, lengths, labels in train_loader:
            binary_padded = binary_padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Feed binary vectors directly into detector (skip _tokens_to_binary)
            # Mask padding
            mask = torch.arange(
                binary_padded.size(1), device=device
            ).unsqueeze(0) < lengths.unsqueeze(1)
            binary_padded = binary_padded * mask.unsqueeze(-1).float()

            preds = discriminator.detector(binary_padded).squeeze(-1)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.get_trainable_params(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        # ── Validate ──
        discriminator.eval()
        all_preds, all_labels_list = [], []
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for binary_padded, lengths, labels in val_loader:
                binary_padded = binary_padded.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                mask = torch.arange(
                    binary_padded.size(1), device=device
                ).unsqueeze(0) < lengths.unsqueeze(1)
                binary_padded = binary_padded * mask.unsqueeze(-1).float()

                preds = discriminator.detector(binary_padded).squeeze(-1)
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

    return discriminator


# ════════════════════════════════════════════════════════════
# Training loop (LLM text mode — original GAN pipeline)
# ════════════════════════════════════════════════════════════

def _train_on_llm_text(
    discriminator: UPVDiscriminatorWrapper,
    config: GANConfig,
    device: str,
) -> UPVDiscriminatorWrapper:
    """
    Train detector on real watermarked text (label=1) vs natural text (label=0).
    This is the original GAN pipeline approach.
    """
    # ── 1) Generate data or load cached ──
    data_cache_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrain_data.json")

    if os.path.exists(data_cache_path):
        print(f"[PretrainD] Loading cached data from {data_cache_path}")
        with open(data_cache_path, 'r') as f:
            data = json.load(f)
        real_texts = data['real_watermarked']
        natural_texts = data['natural']
    else:
        print("[PretrainD] Generating training data with LLM...")
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
        collate_fn=lambda b: collate_fn_token_ids(b, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.disc_pretrain_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_token_ids(b, pad_id),
    )

    params = discriminator.count_params()
    print(f"[PretrainD] UPV Detector — total: {params['total']:,}, "
          f"trainable: {params['trainable']:,}, frozen: {params['frozen']:,}")
    print(f"[PretrainD] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    criterion = nn.BCELoss()

    # FIX 2: Only trainable params (LSTM + FC, NOT binary_classifier)
    optimizer = torch.optim.Adam(
        discriminator.get_trainable_params(),
        lr=config.disc_pretrain_lr,
    )

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
            torch.nn.utils.clip_grad_norm_(discriminator.get_trainable_params(), 1.0)
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

    return discriminator


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

def pretrain_detector(config: GANConfig) -> UPVDiscriminatorWrapper:
    """
    Pre-train UPV Detector.

    FIX 2: binary_classifier is ALWAYS FROZEN. Only LSTM + FC are trained.
    FIX 3: Supports two data modes via config.disc_pretrain_data_mode:
           - "random_tokens": random token sequences + z-score (original repo)
           - "llm_text": real watermarked vs natural text from LLM

    Returns:
        UPVDiscriminatorWrapper ready for adversarial training
    """
    set_seed(42)
    device = config.device

    print("=" * 60)
    print("STEP 2: PRE-TRAINING UPV DETECTOR")
    print(f"  Data mode: {config.disc_pretrain_data_mode}")
    print(f"  Epochs: {config.disc_pretrain_epochs}")
    print(f"  LR: {config.disc_pretrain_lr}")
    print(f"  Samples: {config.disc_pretrain_num_samples}")
    print(f"  Freeze embedding: ALWAYS TRUE")
    print("=" * 60)

    # ── Initialize UPV Detector ──
    # FIX 2: ALWAYS freeze_embedding=True
    discriminator = UPVDiscriminatorWrapper(
        bit_number=config.disc_bit_number,
        detector_weights_path=config.upv_detector_weights,
        freeze_embedding=True,   # FIX 2: ALWAYS frozen
        device=device,
    ).to(device)

    # ── Route to appropriate training mode ──
    data_mode = config.disc_pretrain_data_mode

    if data_mode == "random_tokens":
        # FIX 3: Original repo approach
        random_data = generate_random_token_data(config)
        if random_data is not None:
            discriminator = _train_on_binary_data(discriminator, random_data, config, device)
        else:
            print("[PretrainD] Falling back to llm_text mode...")
            discriminator = _train_on_llm_text(discriminator, config, device)
    else:
        # LLM text approach (original GAN pipeline)
        discriminator = _train_on_llm_text(discriminator, config, device)

    # Embedding is already frozen from init, but verify
    frozen_count = sum(
        1 for p in discriminator.detector.binary_classifier.parameters()
        if not p.requires_grad
    )
    total_bc = sum(
        1 for p in discriminator.detector.binary_classifier.parameters()
    )
    print(f"[PretrainD] Done. binary_classifier: {frozen_count}/{total_bc} params frozen. ✓")

    return discriminator
