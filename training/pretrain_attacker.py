# ============================================================
# pretrain_attacker.py — KGW version
# Step 1: Learn watermark + SFT on spoofer data
# ============================================================

import os, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List

from models.attacker import AttackerLLM, WatermarkLearner
from watermark.kgw_watermark import KGWOracle
from utils.helpers import GANConfig, set_seed, log_metrics, ensure_dir


class SFTDataset(Dataset):
    def __init__(self, sft_data: List[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.labels = []

        for item in sft_data:
            prompt = item['prompt']
            response = item['response']
            prompt_ids = tokenizer(prompt, add_special_tokens=True)['input_ids']
            response_ids = tokenizer(response, add_special_tokens=False)['input_ids']
            combined = prompt_ids + response_ids
            if len(combined) > max_length:
                combined = combined[:max_length]
            labels = [-100] * len(prompt_ids) + response_ids
            if len(labels) > max_length:
                labels = labels[:max_length]
            self.input_ids.append(torch.tensor(combined, dtype=torch.long))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def sft_collate_fn(batch, pad_id=0):
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)
    padded_inputs = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        end = len(ids)
        padded_inputs[i, :end] = ids
        padded_labels[i, :end] = lbl
        attention_mask[i, :end] = 1
    return padded_inputs, padded_labels, attention_mask


def pretrain_attacker(config: GANConfig) -> AttackerLLM:
    set_seed(42)
    device = config.device

    print("=" * 60)
    print("STEP 1: PRE-TRAINING ATTACKER (KGW)")
    print("=" * 60)

    # ── 1) Initialize Attacker ──
    attacker = AttackerLLM(
        model_name=config.llm_name,
        device=device,
        lora_r=config.att_lora_r,
        lora_alpha=config.att_lora_alpha,
        lora_dropout=config.att_lora_dropout,
        lora_target_modules=config.att_lora_target_modules,
    )

    # ── 2) Learning Phase ──
    print("[PretrainG] Phase 1: Learning watermark patterns...")
    learner = WatermarkLearner(
        tokenizer=attacker.tokenizer,
        prevctx_width=config.att_prevctx_width,
    )

    wm_data_path = os.path.join(config.adv_checkpoint_dir, "kgw_learning_wm_texts.json")
    base_data_path = os.path.join(config.adv_checkpoint_dir, "kgw_learning_base_texts.json")

    if os.path.exists(wm_data_path) and os.path.exists(base_data_path):
        print("[PretrainG] Loading cached learning data...")
        with open(wm_data_path, 'r') as f:
            wm_texts = json.load(f)
        with open(base_data_path, 'r') as f:
            base_texts = json.load(f)
    else:
        print("[PretrainG] Generating learning data from KGW Oracle...")
        oracle = KGWOracle(
            model_name=config.llm_name,
            device=device,
            gamma=config.wm_gamma,
            delta=config.wm_delta,
            context_width=config.wm_context_width,
            hash_key=config.wm_hash_key,
        )

        prompts = []
        try:
            with open(config.dataset_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'prompt' in item:
                        prompts.append(item['prompt'])
        except:
            prompts = ["The research findings indicate that"] * config.att_learning_num_queries
        prompts = prompts[:config.att_learning_num_queries]

        wm_texts, base_texts = [], []
        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            wm, _ = oracle.generate_watermarked(batch)
            nat, _ = oracle.generate_unwatermarked(batch)
            wm_texts.extend(wm)
            base_texts.extend(nat)
            if (i // batch_size + 1) % 50 == 0:
                print(f"[PretrainG] Learning data: {len(wm_texts)}/{len(prompts)}")

        ensure_dir(config.adv_checkpoint_dir)
        with open(wm_data_path, 'w') as f:
            json.dump(wm_texts, f)
        with open(base_data_path, 'w') as f:
            json.dump(base_texts, f)
        print(f"[PretrainG] Saved learning data to {config.adv_checkpoint_dir}")

        del oracle
        torch.cuda.empty_cache()

    print(f"[PretrainG] Learning from {len(wm_texts)} WM + {len(base_texts)} base texts...")
    learner.learn_from_watermarked(wm_texts)
    learner.learn_from_baseline(base_texts)
    print(f"[PretrainG] WM counts: {learner.counts_wm.total_counts():,}")
    print(f"[PretrainG] Base counts: {learner.counts_base.total_counts():,}")

    # ── 3) Build Spoofer ──
    vocab_size = attacker.model.config.vocab_size
    static_spoofer = learner.build_spoofer(
        vocab_size, spoofer_strength=config.att_spoofer_strength,
    )
    print(f"[PretrainG] Static Spoofer built (strength={config.att_spoofer_strength})")

    # ── 4) Generate SFT data ──
    sft_cache_path = os.path.join(config.adv_checkpoint_dir, "sft_data.json")

    if os.path.exists(sft_cache_path):
        print(f"[PretrainG] Loading cached SFT data from {sft_cache_path}")
        with open(sft_cache_path, 'r') as f:
            sft_data = json.load(f)
    else:
        print(f"[PretrainG] Generating {config.att_pretrain_num_samples} SFT samples...")
        prompts = []
        try:
            with open(config.dataset_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'prompt' in item:
                        prompts.append(item['prompt'])
        except:
            prompts = ["The research findings indicate that"] * config.att_pretrain_num_samples

        sft_data = []
        batch_size = config.att_pretrain_batch_size
        for i in range(0, config.att_pretrain_num_samples, batch_size):
            batch_prompts = [prompts[j % len(prompts)]
                            for j in range(i, min(i + batch_size, config.att_pretrain_num_samples))]
            texts, _ = attacker.generate(
                batch_prompts,
                max_length=config.att_pretrain_max_length,
                temperature=1.0,
                static_spoofer=static_spoofer,
            )
            for prompt, text in zip(batch_prompts, texts):
                sft_data.append({
                    'prompt': prompt,
                    'response': text.replace(prompt, '').strip(),
                })
            if (i // batch_size + 1) % 50 == 0:
                print(f"[PretrainG] SFT data: {len(sft_data)}/{config.att_pretrain_num_samples}")

        sft_data = sft_data[:config.att_pretrain_num_samples]
        ensure_dir(config.adv_checkpoint_dir)
        with open(sft_cache_path, 'w') as f:
            json.dump(sft_data, f)

    # ── 5) SFT Training ──
    print(f"[PretrainG] Phase 2: SFT on {len(sft_data)} samples...")
    dataset = SFTDataset(sft_data, attacker.tokenizer, max_length=config.att_pretrain_max_length)
    pad_id = attacker.tokenizer.pad_token_id or 0
    dataloader = DataLoader(
        dataset, batch_size=config.att_pretrain_batch_size, shuffle=True,
        collate_fn=lambda b: sft_collate_fn(b, pad_id),
    )

    lora_params = attacker.get_lora_params()
    optimizer = torch.optim.AdamW(lora_params, lr=config.att_pretrain_lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(config.att_pretrain_epochs):
        attacker.model.train()
        total_loss, num_batches = 0.0, 0
        for input_ids, labels, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            outputs = attacker.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        log_metrics({'sft_loss': avg_loss}, epoch)

    # Save
    save_path = os.path.join(config.adv_checkpoint_dir, "attacker_pretrained_lora.pt")
    ensure_dir(config.adv_checkpoint_dir)
    lora_state = {name: module.state_dict() for name, module in attacker.lora_modules.items()}
    torch.save(lora_state, save_path)
    print(f"[PretrainG] Saved LoRA weights to {save_path}")

    attacker._learner = learner
    attacker._static_spoofer = static_spoofer
    print("[PretrainG] Done.")
    return attacker
