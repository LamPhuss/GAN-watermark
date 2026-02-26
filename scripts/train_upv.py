"""
============================================================
train_upv.py — REWRITE matching original THU-BPM repo exactly

5 CRITICAL BUGS FIXED vs previous version:

BUG 1: Generator Architecture WRONG
  OLD: Simple MLP(16→16→1), 1,105 params, hidden_dim=16
  FIX: SubNet(16→64→64) + BinaryClassifier with window combine, ~22K params

BUG 2: Watermark Generation WRONG  
  OLD: generator(context_only) → seed → random partition
  FIX: generator([context, candidate]) → green/red per token individually

BUG 3: Detector Architecture WRONG
  OLD: LSTM(input=1, hidden=64, bidirectional)
  FIX: LSTM(input=64, hidden=128, unidirectional) — original TransformerClassifier

BUG 4: Detector Training Data WRONG
  OLD: 800 real text with Tag labels
  FIX: 10,000 random token sequences with z-score > z_value labels

BUG 5: Training Hyperparams WRONG
  OLD: Generator 50 epochs; Detector 30 epochs, lr=0.001
  FIX: Generator 300 epochs; Detector 80 epochs, lr=0.0005

Usage:
  python scripts/train_upv.py                    # all steps
  python scripts/train_upv.py --step 3 4         # step 3-4 only
  python scripts/train_upv.py --use_pretrained   # use repo weights
============================================================
"""

import os
import sys
import json
import random
import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Architecture — EXACT copy from original model_key.py
# ============================================================

class SubNet(nn.Module):
    """Shared embedding: binary token → 64-dim feature. Output=64, NOT 1."""
    def __init__(self, input_dim, num_layers, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BinaryClassifier(nn.Module):
    """Full generator: (batch, window_size, bit_number) → (batch, 1) green prob."""
    def __init__(self, input_dim, window_size, num_layers, hidden_dim=64):
        super().__init__()
        self.sub_net = SubNet(input_dim, num_layers, hidden_dim)
        self.window_size = window_size
        self.relu = nn.ReLU()
        self.combine_layer = nn.Linear(window_size * hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-1])
        sub_net_output = self.sub_net(x)
        sub_net_output = sub_net_output.view(batch_size, -1)
        combined = self.combine_layer(sub_net_output)
        combined = self.relu(combined)
        output = self.output_layer(combined)
        return self.sigmoid(output)


class TransformerClassifier(nn.Module):
    """Detector: EXACT copy from original detector.py."""
    def __init__(self, bit_number, b_layers, input_dim, hidden_dim, num_classes=1, num_layers=2):
        super().__init__()
        self.binary_classifier = SubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x1 = x.view(batch_size * seq_len, -1)
        features = self.binary_classifier(x1)
        features = features.view(batch_size, seq_len, -1)
        output, _ = self.classifier(features)
        output = self.fc_hidden(output[:, -1, :])
        output = self.sigmoid(output)
        output = self.fc(output)
        return self.sigmoid(output)


# ============================================================
# Helpers
# ============================================================

def int_to_bin_list(n, length=16):
    return [int(b) for b in format(n, 'b').zfill(length)]

def get_value(input_x, model):
    with torch.no_grad():
        return (model(input_x) > 0.5).bool().item()

def max_number(bits):
    return (1 << bits) - 1


# ============================================================
# Step 1: Generate Training Data for Generator
# ============================================================

def generate_generator_data(bit_number, sample_number, output_file, window_size):
    """EXACT copy of original generate_data.py."""
    print("=" * 60)
    print("STEP 1: Generate Training Data for Generator")
    print(f"  bit={bit_number}, window={window_size}, samples={sample_number}")
    print("=" * 60)

    numbers = list(range(1, max_number(bit_number)))
    data = []
    for _ in range(sample_number):
        labels = [0, 1]
        random.shuffle(labels)
        combined = []
        for _ in range(window_size - 1):
            combined.append(int_to_bin_list(random.choice(numbers), bit_number))
        for label in labels:
            combined1 = copy.deepcopy(combined)
            combined1.append(int_to_bin_list(random.choice(numbers), bit_number))
            data.append({"data": combined1, "label": label})

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"  ✓ {len(data)} samples → {output_file}")


# ============================================================
# Step 2: Train Generator
# ============================================================

def train_generator(data_dir, bit_number, model_dir, window_size, layers,
                    epochs=300, lr=0.001, batch_size=32, device="cuda"):
    """EXACT copy of original model_key.py train_model()."""
    print("=" * 60)
    print("STEP 2: Train Generator (BinaryClassifier)")
    print(f"  bit={bit_number}, window={window_size}, layers={layers}, epochs={epochs}")
    print("=" * 60)

    import numpy as np
    features, labels = [], []
    with open(data_dir) as f:
        for line in f:
            entry = json.loads(line)
            features.append(entry['data'])
            labels.append(entry['label'])

    train_data = TensorDataset(
        torch.from_numpy(np.array(features)),
        torch.from_numpy(np.array(labels))
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print(f"  Samples: {len(features)}")

    model = BinaryClassifier(bit_number, window_size, layers).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            outputs = model(inputs.float().to(device))
            loss = criterion(outputs.squeeze(), targets.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "combine_model.pt"))
    torch.save(model.sub_net.state_dict(), os.path.join(model_dir, "sub_net.pt"))
    print(f"  ✓ Saved combine_model.pt + sub_net.pt → {model_dir}")
    return model


# ============================================================
# Step 3: WatermarkEngine (generate train + test data)
# ============================================================

class WatermarkEngine:
    """Matches original watermark_model.py. Key: green/red depends on BOTH context AND candidate."""

    def __init__(self, bit_number, window_size, layers, delta, model_dir, device="cuda"):
        self.bit_number = bit_number
        self.vocab = list(range(1, 2 ** bit_number - 1))
        self.window_size = window_size
        self.delta = delta
        self.cache = {}
        self.device = device

        self.model = BinaryClassifier(bit_number, window_size, layers)
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "combine_model.pt"), map_location=device
        ))
        self.model = self.model.to(device).eval()
        print(f"  ✓ BinaryClassifier loaded")

    def judge_green(self, input_ids, current_number):
        last_nums = input_ids[-(self.window_size - 1):] if self.window_size - 1 > 0 else []
        pair = list(last_nums) + [current_number]
        key = tuple(int(x) for x in pair)
        bin_list = [int_to_bin_list(int(n), self.bit_number) for n in pair]

        if key in self.cache:
            return self.cache[key]
        result = get_value(torch.FloatTensor(bin_list).unsqueeze(0).to(self.device), self.model)
        self.cache[key] = result
        return result

    def random_sample(self, input_ids, is_green):
        last_nums = input_ids[-(self.window_size - 1):] if self.window_size - 1 > 0 else []
        while True:
            number = random.choice(self.vocab)
            pair = list(last_nums) + [number]
            key = tuple(int(x) for x in pair)
            bin_list = [int_to_bin_list(int(n), self.bit_number) for n in pair]
            if key in self.cache:
                result = self.cache[key]
            else:
                result = get_value(torch.FloatTensor(bin_list).unsqueeze(0).to(self.device), self.model)
                self.cache[key] = result
            if is_green and result:
                return number
            elif not is_green and not result:
                return number

    def green_token_mask_and_stats(self, token_ids):
        token_ids = [int(x) for x in token_ids]
        green_mask = []
        for i in range(len(token_ids)):
            if i < self.window_size - 1:
                if self.window_size - 1 > 0:
                    ctx = token_ids[-(self.window_size - 1 - i):] + token_ids[:i]
                else:
                    ctx = []
                green_mask.append(1 if self.judge_green(torch.LongTensor(ctx), token_ids[i]) else 0)
            else:
                ctx = token_ids[i - (self.window_size - 1):i]
                green_mask.append(1 if self.judge_green(torch.LongTensor(ctx), token_ids[i]) else 0)

        gc = sum(green_mask)
        total = len(green_mask)
        from math import sqrt
        z = (gc - total * 0.5) / sqrt(total * 0.25) if total > 0 else 0.0
        return green_mask, gc, z

    def generate_list_with_green_ratio(self, length, green_ratio):
        token_list = (random.sample(self.vocab, self.window_size - 1)
                      if self.window_size - 1 > 0 else random.sample(self.vocab, 1))
        is_green = []
        while len(token_list) < length:
            green = 1 if random.random() < green_ratio else 0
            token = self.random_sample(torch.LongTensor(token_list), green == 1)
            token_list.append(token)
            is_green.append(green)

        # Cyclic labeling for initial tokens
        is_green_append = []
        for i in range(self.window_size - 1):
            tail = token_list[-(self.window_size - 1 - i):]
            head = token_list[:i]
            is_green_append.append(
                1 if self.judge_green(torch.LongTensor(tail + head), token_list[i]) else 0
            )
        is_green = is_green_append + is_green
        return token_list, is_green

    def generate_train_data(self, num_samples, output_dir):
        """Detector training data: random token sequences with z-score labels."""
        from tqdm import tqdm
        train_data = []
        for _ in tqdm(range(num_samples), desc="  Generating detector train data"):
            green_ratio = random.random()
            token_list, is_green = self.generate_list_with_green_ratio(200, green_ratio)
            _, _, z_score = self.green_token_mask_and_stats(token_list)
            train_data.append((tuple(token_list), tuple(is_green), z_score))

        train_data = list(set(train_data))
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'train_data.jsonl')
        with open(path, 'w') as f:
            for item in train_data:
                json.dump({"Input": list(item[0]), "Tag": list(item[1]), "Output": item[2]}, f)
                f.write('\n')
        print(f"  ✓ {len(train_data)} train samples → {path}")
        return path

    def generate_test_data(self, llm_name, output_dir, data_path=None,
                           sampling_temp=0.7, max_new_tokens=200, num_samples=500):
        """Test data: real watermarked + natural text using LLM."""
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  LogitsProcessor, LogitsProcessorList)

        print(f"  Loading LLM: {llm_name}...")
        device = self.device
        tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=False)
        load_kwargs = {"torch_dtype": torch.float16}
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("  [FlashAttn] Using flash_attention_2")
        except ImportError:
            pass
        model = AutoModelForCausalLM.from_pretrained(llm_name, **load_kwargs).to(device)

        engine = self

        class WMProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if input_ids.shape[-1] < engine.window_size - 1:
                    return scores
                for b in range(input_ids.shape[0]):
                    ids = input_ids[b]
                    if engine.window_size - 1 > 0:
                        last = ids[-(engine.window_size - 1):].cpu().tolist()
                    else:
                        last = []
                    _, cands = torch.topk(scores[b], k=min(20, scores.shape[-1]))
                    greens = []
                    for v in cands:
                        pair = last + [int(v)]
                        key = tuple(int(x) for x in pair)
                        bins = [int_to_bin_list(int(n), engine.bit_number) for n in pair]
                        if key in engine.cache:
                            r = engine.cache[key]
                        else:
                            r = get_value(torch.FloatTensor(bins).unsqueeze(0).to(device), engine.model)
                            engine.cache[key] = r
                        if r:
                            greens.append(int(v))
                    if greens:
                        scores[b][greens] += engine.delta
                    if "opt" in llm_name:
                        scores[b][2] = -10000
                return scores

        prompts = _load_prompts(tokenizer, num_samples, data_path)
        wm_out, nat_out = [], []

        for i, prompt_text in enumerate(prompts):
            try:
                inp = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
                plen = inp["input_ids"].shape[-1]

                with torch.no_grad():
                    out_wm = model.generate(
                        **inp, logits_processor=LogitsProcessorList([WMProcessor()]),
                        max_new_tokens=max_new_tokens, do_sample=True,
                        top_k=20, temperature=sampling_temp, no_repeat_ngram_size=4,
                    )
                gen_wm = out_wm[0, plen:].cpu().tolist()
                _, _, z_wm = self.green_token_mask_and_stats(gen_wm)
                wm_out.append({"Input": tokenizer.decode(gen_wm, skip_special_tokens=True),
                               "Tag": 1, "Z-score": z_wm})

                with torch.no_grad():
                    out_nat = model.generate(
                        **inp, max_new_tokens=max_new_tokens, do_sample=True,
                        top_k=20, temperature=sampling_temp,
                    )
                gen_nat = out_nat[0, plen:].cpu().tolist()
                _, _, z_nat = self.green_token_mask_and_stats(gen_nat)
                nat_out.append({"Input": tokenizer.decode(gen_nat, skip_special_tokens=True),
                                "Tag": 0, "Z-score": z_nat})

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{num_samples} (wm z={z_wm:.2f}, nat z={z_nat:.2f})")
            except Exception as e:
                print(f"  ⚠ Sample {i}: {e}")

        path = os.path.join(output_dir, 'test_data.jsonl')
        with open(path, 'w') as f:
            for item in wm_out + nat_out:
                json.dump(item, f)
                f.write('\n')
        print(f"  ✓ {len(wm_out)} wm + {len(nat_out)} nat → {path}")
        del model; torch.cuda.empty_cache()
        return path


def _load_prompts(tokenizer, num_samples, data_path=None):
    if data_path and os.path.exists(data_path):
        prompts = []
        with open(data_path) as f:
            content = f.read()
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(content) and len(prompts) < num_samples:
            s = content[idx:].lstrip()
            if not s: break
            if s[0] in '[],':
                idx += len(content[idx:]) - len(s) + 1; continue
            try:
                obj, end = decoder.raw_decode(s)
                idx += len(content[idx:]) - len(s) + end
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                if "prompt" in obj: prompts.append(obj["prompt"])
                elif "text" in obj:
                    toks = tokenizer(obj["text"], add_special_tokens=False)["input_ids"]
                    if len(toks) > 30: prompts.append(tokenizer.decode(toks[:30]))
        if prompts:
            print(f"  ✓ {len(prompts)} prompts from {data_path}")
            return prompts
    print("  Using fallback prompts")
    base = ["The latest research in artificial intelligence suggests that",
            "In a groundbreaking study, scientists discovered that",
            "The economic impact of climate change has been",
            "According to recent findings in neuroscience,",
            "New regulations aimed at reducing carbon emissions",
            "The relationship between diet and mental health",
            "Advances in renewable energy technology have",
            "The global supply chain disruptions caused by"]
    return [base[i % len(base)] for i in range(num_samples)]


# ============================================================
# Step 4: Train Detector
# ============================================================

def pad_to_fixed(inputs, target_length):
    padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    L = padded.shape[1]
    if L < target_length:
        padded = F.pad(padded, (0, 0, 0, target_length - L))
    elif L > target_length:
        padded = padded[:, :target_length, :]
    return padded

def train_collate(batch):
    return pad_to_fixed([b[0] for b in batch], 200), torch.stack([b[1] for b in batch])

def test_collate(batch):
    return (pad_to_fixed([b[0] for b in batch], 200),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


def train_detector(bit_number, input_dir, model_file, output_model_dir,
                   b_layers, z_value, llm_name, epochs=80, lr=0.0005, device="cuda"):
    """EXACT copy of original detector.py."""
    print("=" * 60)
    print("STEP 4: Train Detector (TransformerClassifier)")
    print(f"  epochs={epochs}, lr={lr}, z_value={z_value}")
    print("=" * 60)

    # Train data: random token sequences, label = z-score > z_value
    train_data = []
    with open(os.path.join(input_dir, 'train_data.jsonl')) as f:
        for line in f:
            obj = json.loads(line)
            bins = [int_to_bin_list(n, bit_number) for n in obj['Input']]
            label = 1 if obj['Output'] > z_value else 0
            train_data.append((torch.tensor(bins), torch.tensor(label)))
    print(f"  Train: {len(train_data)} samples")

    # Test data: real text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=False)
    test_data = []
    with open(os.path.join(input_dir, 'test_data.jsonl')) as f:
        for line in f:
            obj = json.loads(line)
            toks = tokenizer(obj['Input'], return_tensors="pt", add_special_tokens=True)
            bins = [int_to_bin_list(int(n), bit_number) for n in toks["input_ids"].squeeze()]
            test_data.append((torch.tensor(bins), torch.tensor(obj['Tag']),
                              torch.tensor(obj['Z-score'])))
    print(f"  Test: {len(test_data)} samples")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=train_collate)
    test_loader = DataLoader(test_data, batch_size=32, collate_fn=test_collate)

    # Model
    model = TransformerClassifier(bit_number, b_layers, 64, 128).to(device)

    # Load shared embedding
    pretrained = torch.load(model_file, map_location=device)
    model_dict = model.binary_classifier.state_dict()
    pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained)
    model.binary_classifier.load_state_dict(model_dict, strict=True)
    for p in model.binary_classifier.parameters():
        p.requires_grad = False
    print(f"  ✓ Shared embedding loaded & FROZEN")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        losses, correct, total_n = [], 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.float().to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).reshape([-1])
            loss = loss_fn(outputs, targets.float())
            loss.backward(); optimizer.step()
            losses.append(loss.item())
            correct += ((outputs > 0.5).float() == targets).sum().item()
            total_n += targets.size(0)

        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs, targets = inputs.float().to(device), targets.to(device)
                pred = (model(inputs).reshape([-1]) > 0.5).int()
                tp += (pred & targets).sum().item()
                fp += (pred & (~targets.bool())).sum().item()
                fn += ((~pred.bool()) & targets).sum().item()
                tn += ((~pred.bool()) & (~targets.bool())).sum().item()

        acc = 100 * (tp + tn) / max(tp + fp + fn + tn, 1)
        f1 = 100 * 2 * tp / max(2 * tp + fn + fp, 1)
        tpr = 100 * tp / max(tp + fn, 1)
        fpr = 100 * fp / max(fp + tn, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {sum(losses)/len(losses):.4f} | "
                  f"Acc: {acc:.1f}% | F1: {f1:.1f}% | TPR: {tpr:.1f}% | FPR: {fpr:.1f}%")

    os.makedirs(output_model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_model_dir, "detector_model.pt"))
    torch.save(model.binary_classifier.state_dict(), os.path.join(output_model_dir, "detector_subnet.pt"))
    print(f"  ✓ Detector saved → {output_model_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--llm", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--bit_number", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--z_value", type=float, default=4.0)
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_test_samples", type=int, default=500)
    parser.add_argument("--data_path", type=str, default="data/processed_c4.jsonl")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "upv"))
    args = parser.parse_args()

    if args.data_path and not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)

    model_dir = os.path.join(args.output_dir, "model")
    data_dir = os.path.join(args.output_dir, "data")
    train_data_dir = os.path.join(args.output_dir, "train_data")
    for d in [model_dir, data_dir, train_data_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("UPV Training Pipeline (ORIGINAL architecture)")
    print(f"  Steps: {args.step}")
    print(f"  bit={args.bit_number}, window={args.window_size}, layers={args.layers}")
    print(f"  Device: {args.device}")
    if args.data_path:
        print(f"  Data: {args.data_path} (exists: {os.path.exists(args.data_path)})")
    print("=" * 60)

    # Handle pre-trained
    if args.use_pretrained:
        repo = os.path.join(PROJECT_ROOT, "unforgeable_watermark")
        paths = [
            os.path.join(repo, f"experiments/robustness/generator_model/opt-1.3b/window_size_{args.window_size}"),
            os.path.join(repo, "experiments/main_experiments/generator_model"),
        ]
        found = False
        for p in paths:
            if os.path.exists(os.path.join(p, "combine_model.pt")):
                import shutil
                shutil.copy(os.path.join(p, "combine_model.pt"), os.path.join(model_dir, "combine_model.pt"))
                shutil.copy(os.path.join(p, "sub_net.pt"), os.path.join(model_dir, "sub_net.pt"))
                print(f"  ✓ Pre-trained weights from {p}")
                found = True; break
        if not found:
            print("  ⚠ Not found. Run: git clone https://github.com/THU-BPM/unforgeable_watermark.git")
            return
        args.step = [s for s in args.step if s >= 3]

    t0 = time.time()
    gen_data = os.path.join(train_data_dir, "train_generator_data.jsonl")

    if 1 in args.step:
        generate_generator_data(args.bit_number, 2000, gen_data, args.window_size)

    if 2 in args.step:
        train_generator(gen_data, args.bit_number, model_dir, args.window_size,
                        args.layers, epochs=300, device=args.device)

    if 3 in args.step:
        engine = WatermarkEngine(args.bit_number, args.window_size, args.layers,
                                 args.delta, model_dir, args.device)
        print("=" * 60)
        print("STEP 3a: Detector Training Data (random token sequences)")
        print("=" * 60)
        engine.generate_train_data(args.num_train_samples, data_dir)

        print("=" * 60)
        print("STEP 3b: Test Data (watermarked + natural text)")
        print("=" * 60)
        engine.generate_test_data(args.llm, data_dir, args.data_path,
                                  num_samples=args.num_test_samples)

    if 4 in args.step:
        train_detector(args.bit_number, data_dir, os.path.join(model_dir, "sub_net.pt"),
                       model_dir, args.layers, args.z_value, args.llm,
                       epochs=80, lr=0.0005, device=args.device)

    print(f"\nDone! {(time.time()-t0)/60:.1f} minutes")
    print("Model files:")
    for f in sorted(os.listdir(model_dir)):
        print(f"  {f} ({os.path.getsize(os.path.join(model_dir,f))/1024:.1f} KB)")

if __name__ == "__main__":
    main()
