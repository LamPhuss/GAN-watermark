"""
============================================================
train_upv.py
Train UPV Generator + Detector from scratch.

Produces:
  - upv/model/generator_model_b16_p1.pt  (Generator / UPVSubNet)
  - upv/model/detector_model_b16_p1_z4.pt (Detector)

Pipeline overview:
  Step 1: Generate random binary training data (NO LLM needed)
  Step 2: Train Generator network (~43K params, CPU-friendly)
  Step 3: Generate watermarked text using LLM + Generator (GPU)
  Step 4: Train Detector using shared embedding from Generator (GPU)

Usage:
  python scripts/train_upv.py                    # all 4 steps
  python scripts/train_upv.py --step 1           # only step 1
  python scripts/train_upv.py --step 2           # only step 2
  python scripts/train_upv.py --step 3 4         # steps 3 and 4
  python scripts/train_upv.py --llm opt-1.3b     # specify LLM
============================================================
"""

import os
import sys
import json
import random
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# UPV Network Architecture (from the paper)
# ============================================================

class UPVSubNet(nn.Module):
    """
    UPV Generator Network.
    
    Input:  binary vector (window_size * bit_number,)
    Output: probability of token being "green" (1,)
    
    Architecture: MLP with skip connections
      Input → FC → ReLU → [FC → ReLU] × (layers-2) → FC → Sigmoid
    
    The paper uses this to split vocabulary into green/red lists
    based on preceding token context.
    """

    def __init__(self, bit_number: int = 16, window_size: int = 1, layers: int = 5):
        super().__init__()
        input_dim = bit_number * window_size
        hidden_dim = bit_number * window_size  # Same as input

        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.ReLU())

        for _ in range(layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(hidden_dim, 1))
        modules.append(nn.Sigmoid())

        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UPVDetectorNet(nn.Module):
    """
    UPV Detector Network.
    
    Input:  sequence of binary vectors (seq_len, bit_number)
    Output: probability of sequence being watermarked (1,)
    
    Architecture: 
      binary_classifier (shared from Generator) → LSTM → FC → Sigmoid
    
    CRITICAL: binary_classifier weights come from the trained Generator
    and should NOT be fine-tuned (paper shows -11.1% F1 if fine-tuned).
    """

    def __init__(
        self,
        bit_number: int = 16,
        hidden_size: int = 64,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        self.bit_number = bit_number

        # Shared embedding from Generator (will be loaded later)
        self.binary_classifier = None  # Set after generator training

        # LSTM processes sequence of embeddings
        self.lstm = nn.LSTM(
            input_size=1,  # Output of binary_classifier per token
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def set_shared_embedding(self, generator: UPVSubNet):
        """Copy generator network as shared embedding (frozen)."""
        self.binary_classifier = generator
        # FREEZE shared embedding
        for param in self.binary_classifier.parameters():
            param.requires_grad = False
        print("[Detector] Shared embedding set and FROZEN")

    def forward(self, binary_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            binary_seq: (batch, seq_len, bit_number)
        Returns:
            (batch, 1) watermark probability
        """
        batch_size, seq_len, bit_num = binary_seq.shape

        # Apply shared embedding to each token
        flat = binary_seq.reshape(-1, bit_num)  # (batch*seq, bit_num)
        with torch.no_grad():  # Shared embedding is frozen
            token_scores = self.binary_classifier(flat)  # (batch*seq, 1)
        token_scores = token_scores.reshape(batch_size, seq_len, 1)

        # LSTM over sequence
        lstm_out, _ = self.lstm(token_scores)  # (batch, seq, hidden*2)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden*2)

        # Classify
        prob = self.fc(last_hidden)  # (batch, 1)
        return prob


# ============================================================
# Step 1: Generate Training Data
# ============================================================

def int_to_bin(val: int, bit_number: int) -> list:
    """Convert integer to binary list (MSB first)."""
    return [(val >> (bit_number - 1 - i)) & 1 for i in range(bit_number)]


def generate_training_data(
    bit_number: int = 16,
    window_size: int = 1,
    sample_number: int = 2000,
    seq_length: int = 200,
    output_file: str = "upv/train_data/train_generator_data.jsonl",
):
    """
    Step 1: Generate random binary training data for Generator.
    
    Each sample is a random token ID sequence. For each position,
    we extract the context window (preceding tokens in binary) and
    assign a random green/red label.
    
    The Generator will learn to produce ~50% green tokens for any
    input context, creating a balanced green/red split.
    
    NO LLM needed — purely random data.
    """
    print("=" * 60)
    print("STEP 1: Generate Training Data for Generator")
    print(f"  bit_number: {bit_number}")
    print(f"  window_size: {window_size}")
    print(f"  samples: {sample_number}")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    max_val = 2 ** bit_number

    data = []
    for i in range(sample_number):
        # Random token ID sequence
        seq = [random.randint(0, max_val - 1) for _ in range(seq_length)]

        # For each position, create (context_binary, label) pairs
        for pos in range(window_size, len(seq)):
            # Context: preceding `window_size` tokens as binary
            ctx_tokens = seq[pos - window_size : pos]
            ctx_binary = []
            for t in ctx_tokens:
                ctx_binary.extend(int_to_bin(t, bit_number))

            # Current token binary
            cur_binary = int_to_bin(seq[pos], bit_number)

            # Target: we want ~50% green, so label = hash-based assignment
            # Simple approach: sum of context bits mod 2
            label = sum(ctx_binary) % 2

            data.append({
                "context": ctx_binary,
                "token": cur_binary,
                "label": label,
            })

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{sample_number} sequences ({len(data)} samples)")

    # Save
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"  ✓ Saved {len(data)} training samples to {output_file}")
    return output_file


# ============================================================
# Step 2: Train Generator
# ============================================================

class GeneratorDataset(Dataset):
    def __init__(self, data_file: str):
        self.samples = []
        with open(data_file) as f:
            for line in f:
                item = json.loads(line)
                self.samples.append((
                    torch.tensor(item["context"], dtype=torch.float32),
                    torch.tensor([item["label"]], dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_generator(
    data_file: str,
    bit_number: int = 16,
    window_size: int = 1,
    layers: int = 5,
    output_dir: str = "upv/model",
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 256,
    device: str = "cpu",
):
    """
    Step 2: Train Generator network (UPVSubNet).
    
    This is a tiny network (~43K params). Trains in minutes on CPU.
    
    The Generator learns to assign green/red labels to tokens based
    on context. After training, it should produce ~50% green tokens
    for any random context (balanced split).
    
    Output: generator_model_b16_p1.pt (= sub_net.pt in original repo)
    """
    print("=" * 60)
    print("STEP 2: Train Generator Network (UPVSubNet)")
    print(f"  bit_number: {bit_number}, window_size: {window_size}")
    print(f"  layers: {layers}, epochs: {epochs}")
    print(f"  device: {device}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    dataset = GeneratorDataset(data_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  Training samples: {len(dataset)}")

    # Model
    model = UPVSubNet(bit_number=bit_number, window_size=window_size, layers=layers)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Generator params: {total_params:,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for ctx, label in loader:
            ctx, label = ctx.to(device), label.to(device)
            pred = model(ctx)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * ctx.size(0)
            correct += ((pred > 0.5).float() == label).sum().item()
            total += ctx.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(output_dir, "generator_model_b16_p1.pt")
            torch.save(model.state_dict(), save_path)

    save_path = os.path.join(output_dir, "generator_model_b16_p1.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Generator saved to {save_path}")
    print(f"  ✓ Best loss: {best_loss:.4f}")

    return model


# ============================================================
# Step 3: Generate Watermarked Text Data
# ============================================================

def generate_watermarked_data(
    generator_path: str,
    llm_name: str = "facebook/opt-1.3b",
    bit_number: int = 16,
    window_size: int = 1,
    delta: float = 2.0,
    num_samples: int = 500,
    max_new_tokens: int = 200,
    output_dir: str = "upv/data",
    data_path: str = None,
    device: str = "cuda",
):
    """
    Step 3: Generate watermarked + unwatermarked text.
    
    Uses the trained Generator to modify LLM logits during generation:
    - Green tokens get +delta boost to their logits
    - Red tokens remain unchanged
    
    This creates the distribution shift that makes watermarked
    text detectable.
    
    REQUIRES GPU for LLM inference.
    """
    print("=" * 60)
    print("STEP 3: Generate Watermarked Text Data")
    print(f"  LLM: {llm_name}")
    print(f"  delta: {delta}, samples: {num_samples}")
    print(f"  device: {device}")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load Generator
    generator = UPVSubNet(bit_number=bit_number, window_size=window_size)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.to(device)
    generator.eval()
    print(f"  ✓ Generator loaded from {generator_path}")

    # Load LLM
    print(f"  Loading LLM: {llm_name}...")
    load_kwargs = {"torch_dtype": torch.float16}
    try:
        import flash_attn
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("  [FlashAttn] Using flash_attention_2")
    except ImportError:
        print("  [FlashAttn] Not available, using default")

    model = AutoModelForCausalLM.from_pretrained(llm_name, **load_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load prompts
    prompts = _load_c4_prompts(tokenizer, num_samples, data_path=data_path)

    # Generate
    watermarked_data = []
    unwatermarked_data = []

    for i, prompt in enumerate(prompts):
        # Watermarked generation
        wm_text, wm_z = _generate_single(
            model, tokenizer, generator, prompt,
            bit_number, window_size, delta, max_new_tokens, device,
            watermark=True,
        )
        watermarked_data.append({"Input": wm_text, "Tag": 1, "Z-score": wm_z})

        # Unwatermarked generation (same prompt)
        nat_text, nat_z = _generate_single(
            model, tokenizer, generator, prompt,
            bit_number, window_size, delta, max_new_tokens, device,
            watermark=False,
        )
        unwatermarked_data.append({"Input": nat_text, "Tag": 0, "Z-score": nat_z})

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} pairs")

    # Save
    train_path = os.path.join(output_dir, "train_data.jsonl")
    test_path = os.path.join(output_dir, "test_data.jsonl")

    # 80/20 split
    split = int(0.8 * len(watermarked_data))
    all_data = watermarked_data + unwatermarked_data
    random.shuffle(all_data)

    with open(train_path, "w") as f:
        for item in all_data[:split * 2]:
            f.write(json.dumps(item) + "\n")

    with open(test_path, "w") as f:
        for item in all_data[split * 2:]:
            f.write(json.dumps(item) + "\n")

    print(f"  ✓ Train data: {train_path} ({split * 2} samples)")
    print(f"  ✓ Test data: {test_path} ({len(all_data) - split * 2} samples)")

    # Cleanup LLM to free VRAM
    del model
    torch.cuda.empty_cache()

    return train_path, test_path


def _load_c4_prompts(tokenizer, num_samples: int, data_path: str = None) -> list:
    """Load prompts from user's processed_c4.json or fallback to HuggingFace C4."""
    
    # Priority 1: User's processed_c4.json
    if data_path and os.path.exists(data_path):
        prompts = []
        try:
            with open(data_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    if "prompt" in item:
                        prompts.append(item["prompt"])
                    elif "text" in item:
                        # Extract first 30 tokens as prompt
                        tokens = tokenizer(item["text"], add_special_tokens=False)["input_ids"]
                        if len(tokens) > 30:
                            prompts.append(tokenizer.decode(tokens[:30]))
                    if len(prompts) >= num_samples:
                        break
            print(f"  ✓ Loaded {len(prompts)} prompts from {data_path}")
            return prompts
        except Exception as e:
            print(f"  ⚠ Error reading {data_path}: {e}")
    
    # Priority 2: HuggingFace C4
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        prompts = []
        for item in ds:
            text = item["text"]
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(tokens) > 30:
                prompt = tokenizer.decode(tokens[:30])
                prompts.append(prompt)
            if len(prompts) >= num_samples:
                break
        return prompts
    except Exception as e:
        print(f"  ⚠ Could not load C4: {e}. Using fallback prompts.")
        base = [
            "The latest research in artificial intelligence suggests that",
            "In a groundbreaking study, scientists discovered that",
            "The economic impact of climate change has been",
            "According to recent findings in neuroscience,",
            "New regulations aimed at reducing carbon emissions",
            "The relationship between diet and mental health",
            "Advances in renewable energy technology have",
            "The global supply chain disruptions caused by",
        ]
        return [base[i % len(base)] for i in range(num_samples)]


@torch.no_grad()
def _generate_single(
    model, tokenizer, generator, prompt,
    bit_number, window_size, delta, max_new_tokens, device,
    watermark=True,
):
    """Generate a single text with or without watermark."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    max_val = 2 ** bit_number

    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # (1, vocab)

        if watermark:
            # Get context tokens for generator
            ctx_start = max(0, input_ids.size(1) - window_size)
            ctx_tokens = input_ids[0, ctx_start:].tolist()

            # For each vocab token, check if it's "green"
            vocab_size = logits.size(-1)
            for tok_id in range(min(vocab_size, max_val)):
                ctx_binary = []
                for t in ctx_tokens:
                    ctx_binary.extend(int_to_bin(t % max_val, bit_number))
                # Pad if context shorter than window_size
                while len(ctx_binary) < bit_number * window_size:
                    ctx_binary = [0] * bit_number + ctx_binary

                tok_binary = int_to_bin(tok_id, bit_number)
                full_input = ctx_binary + tok_binary

                # Check generator output
                inp = torch.tensor(
                    ctx_binary[-bit_number * window_size:],
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                green_prob = generator(inp).item()

                if green_prob > 0.5:
                    logits[0, tok_id] += delta  # Boost green tokens

        # Sample
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated = input_ids[0, inputs["input_ids"].size(1):]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    # Compute z-score
    z_score = _compute_z_score(generator, generated, bit_number, window_size, device)

    return text, z_score


def _compute_z_score(
    generator, token_ids, bit_number, window_size, device,
) -> float:
    """Compute z-score: how many tokens are 'green'."""
    max_val = 2 ** bit_number
    green_count = 0
    total = 0

    for i in range(window_size, len(token_ids)):
        ctx_start = max(0, i - window_size)
        ctx_tokens = token_ids[ctx_start:i].tolist()

        ctx_binary = []
        for t in ctx_tokens:
            ctx_binary.extend(int_to_bin(t % max_val, bit_number))
        while len(ctx_binary) < bit_number * window_size:
            ctx_binary = [0] * bit_number + ctx_binary

        inp = torch.tensor(
            ctx_binary[-bit_number * window_size:],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

        green_prob = generator(inp).item()
        if green_prob > 0.5:
            green_count += 1
        total += 1

    if total == 0:
        return 0.0

    # z = (green_count - T/2) / sqrt(T/4)
    expected = total * 0.5
    std = (total * 0.25) ** 0.5
    z = (green_count - expected) / (std + 1e-8)
    return z


# ============================================================
# Step 4: Train Detector
# ============================================================

def train_detector(
    generator_path: str,
    train_data_path: str,
    llm_name: str = "facebook/opt-1.3b",
    bit_number: int = 16,
    window_size: int = 1,
    z_value: float = 4.0,
    output_dir: str = "upv/model",
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Step 4: Train Detector network.
    
    The Detector shares its binary_classifier (embedding layer)
    with the Generator. This shared embedding is FROZEN — only
    the LSTM and FC layers are trained.
    
    Training data: watermarked text (label=1) vs natural text (label=0)
    """
    print("=" * 60)
    print("STEP 4: Train Detector Network")
    print(f"  generator: {generator_path}")
    print(f"  z_value threshold: {z_value}")
    print(f"  device: {device}")
    print("=" * 60)

    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load Generator (for shared embedding)
    generator = UPVSubNet(bit_number=bit_number, window_size=window_size)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.to(device)
    generator.eval()

    # Build Detector with shared embedding
    detector = UPVDetectorNet(bit_number=bit_number)
    detector.set_shared_embedding(generator)
    detector.to(device)

    trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    total = sum(p.numel() for p in detector.parameters())
    print(f"  Detector params: {total:,} total, {trainable:,} trainable")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    # Load training data
    texts, labels = [], []
    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["Input"])
            labels.append(item["Tag"])

    print(f"  Training samples: {len(texts)} ({sum(labels)} watermarked, {len(labels) - sum(labels)} natural)")

    # Convert to binary sequences
    max_val = 2 ** bit_number
    max_seq_len = 200

    binary_seqs = []
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_seq_len]
        binary = []
        for tid in token_ids:
            binary.append(int_to_bin(tid % max_val, bit_number))
        # Pad
        while len(binary) < max_seq_len:
            binary.append([0] * bit_number)
        binary_seqs.append(binary[:max_seq_len])

    X = torch.tensor(binary_seqs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    dataset = torch.utils.data.TensorDataset(X, y)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        [p for p in detector.parameters() if p.requires_grad],
        lr=lr,
    )

    best_f1 = 0.0
    for epoch in range(epochs):
        # Train
        detector.train()
        train_loss = 0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = detector(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        # Validate
        detector.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                pred = detector(X_batch)
                all_preds.extend((pred > 0.5).float().cpu().squeeze().tolist())
                all_labels.extend(y_batch.squeeze().tolist())

        # F1 score
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {train_loss / n_batches:.4f} | "
                  f"F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(output_dir, "detector_model_b16_p1_z4.pt")
            torch.save(detector.state_dict(), save_path)

    save_path = os.path.join(output_dir, "detector_model_b16_p1_z4.pt")
    torch.save(detector.state_dict(), save_path)
    print(f"  ✓ Detector saved to {save_path}")
    print(f"  ✓ Best F1: {best_f1:.4f}")

    return detector


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train UPV Generator + Detector")
    parser.add_argument("--step", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which steps to run (1-4)")
    parser.add_argument("--llm", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--bit_number", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of watermarked text samples to generate")
    parser.add_argument("--data_path", type=str, default="data/processed_c4.json",
                        help="Path to your processed_c4.json with prompts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "upv"))

    args = parser.parse_args()

    model_dir = os.path.join(args.output_dir, "model")
    data_dir = os.path.join(args.output_dir, "data")
    train_data_dir = os.path.join(args.output_dir, "train_data")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)

    print("=" * 60)
    print("UPV Training Pipeline")
    print(f"  Steps: {args.step}")
    print(f"  LLM: {args.llm}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    train_data_file = os.path.join(train_data_dir, "train_generator_data.jsonl")
    generator_path = os.path.join(model_dir, "generator_model_b16_p1.pt")
    wm_train_path = os.path.join(data_dir, "train_data.jsonl")

    t0 = time.time()

    if 1 in args.step:
        generate_training_data(
            bit_number=args.bit_number,
            window_size=args.window_size,
            sample_number=2000,
            output_file=train_data_file,
        )

    if 2 in args.step:
        train_generator(
            data_file=train_data_file,
            bit_number=args.bit_number,
            window_size=args.window_size,
            output_dir=model_dir,
            device=args.device,
        )

    if 3 in args.step:
        wm_train_path, _ = generate_watermarked_data(
            generator_path=generator_path,
            llm_name=args.llm,
            bit_number=args.bit_number,
            window_size=args.window_size,
            delta=args.delta,
            num_samples=args.num_samples,
            output_dir=data_dir,
            data_path=args.data_path,
            device=args.device,
        )

    if 4 in args.step:
        train_detector(
            generator_path=generator_path,
            train_data_path=wm_train_path,
            llm_name=args.llm,
            bit_number=args.bit_number,
            window_size=args.window_size,
            output_dir=model_dir,
            device=args.device,
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")
    print(f"\nModel files:")
    for f in os.listdir(model_dir):
        fpath = os.path.join(model_dir, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
