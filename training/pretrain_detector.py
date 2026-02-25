# ============================================================
# pretrain_detector.py
# Description: Step 2 — Pre-train the LSTM Discriminator (D_φ)
#   on real watermarked vs natural text
# ============================================================

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score

from models.detector import WatermarkDiscriminator
from data.data_generator import UPVOracle, DataGenerator, pad_sequences
from utils.helpers import GANConfig, set_seed, log_metrics, ensure_dir


class WatermarkDataset(Dataset):
    """
    Dataset for Discriminator pre-training.
    
    Each sample is (token_ids, label):
        - label=1: Real watermarked text (from UPV Generator)
        - label=0: Natural text (no watermark)
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts
        self.token_ids = []
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=max_length, add_special_tokens=False)["input_ids"][0]
            self.token_ids.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


def collate_fn(batch, pad_value=0):
    """Collate function for variable-length sequences."""
    token_ids_list, labels = zip(*batch)
    
    padded, lengths = pad_sequences(list(token_ids_list), pad_value=pad_value)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return padded, lengths, labels


def pretrain_detector(config: GANConfig) -> WatermarkDiscriminator:
    """
    Pre-train the LSTM Discriminator on real watermarked vs natural text.
    
    Steps:
        1. Use UPV Generator to create N watermarked texts (label=1)
        2. Use base LLM to create N natural texts (label=0)
        3. Train LSTM Discriminator until F1 > target_f1
    
    Returns:
        Trained WatermarkDiscriminator
    """
    set_seed(42)
    device = config.device
    
    print("=" * 60)
    print("STEP 2: PRE-TRAINING DISCRIMINATOR")
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
    
    # Combine real (label=1) and natural (label=0)
    all_texts = real_texts + natural_texts
    all_labels = [1] * len(real_texts) + [0] * len(natural_texts)
    
    # Shuffle
    import random
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    
    # Split 80/20
    split_idx = int(0.8 * len(all_texts))
    train_dataset = WatermarkDataset(list(all_texts[:split_idx]), list(all_labels[:split_idx]), tokenizer)
    val_dataset = WatermarkDataset(list(all_texts[split_idx:]), list(all_labels[split_idx:]), tokenizer)
    
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
    
    # ── 3) Initialize Discriminator ──
    vocab_size = tokenizer.vocab_size
    discriminator = WatermarkDiscriminator(
        vocab_size=vocab_size,
        embedding_dim=config.disc_embedding_dim,
        hidden_dim=config.disc_hidden_dim,
        num_lstm_layers=config.disc_num_lstm_layers,
        dropout=config.disc_dropout,
        freeze_embedding=False,  # Train embedding during pretraining
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.disc_pretrain_lr)
    
    print(f"[PretrainD] Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"[PretrainD] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # ── 4) Training loop ──
    best_f1 = 0.0
    
    for epoch in range(config.disc_pretrain_epochs):
        # Training
        discriminator.train()
        train_loss = 0.0
        num_batches = 0
        
        for padded, lengths, labels in train_loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = discriminator(padded, lengths).squeeze(-1)  # (batch,)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        
        # Validation
        discriminator.eval()
        all_preds = []
        all_labels = []
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
                
                binary_preds = (preds > 0.5).float()
                all_preds.extend(binary_preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        avg_val_loss = val_loss / max(val_batches, 1)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }
        log_metrics(metrics, epoch)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(config.adv_checkpoint_dir, "disc_pretrained_best.pt")
            ensure_dir(config.adv_checkpoint_dir)
            torch.save(discriminator.state_dict(), save_path)
            print(f"[PretrainD] Saved best model (F1={f1:.4f}) to {save_path}")
        
        # Early stop if target reached
        if f1 >= config.disc_target_f1:
            print(f"[PretrainD] Target F1 ({config.disc_target_f1}) reached at epoch {epoch}!")
            break
    
    # ── 5) FREEZE embedding after pre-training ──
    discriminator.freeze_embedding()
    print(f"[PretrainD] Done. Best F1: {best_f1:.4f}. Embedding FROZEN.")
    
    return discriminator
