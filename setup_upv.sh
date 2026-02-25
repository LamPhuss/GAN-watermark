#!/bin/bash
# ============================================================
# setup_upv.sh
# Download pre-trained UPV weights from official repo
# OR train from scratch
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UPV_MODEL_DIR="$PROJECT_DIR/upv/model"
UPV_REPO_DIR="$PROJECT_DIR/unforgeable_watermark"

mkdir -p "$UPV_MODEL_DIR"

echo "============================================================"
echo "UPV Model Setup"
echo "============================================================"

# ─── Option 1: Clone repo and copy pre-trained weights ───
if [ ! -d "$UPV_REPO_DIR" ]; then
    echo "[1/4] Cloning UPV repo..."
    git clone https://github.com/THU-BPM/unforgeable_watermark.git "$UPV_REPO_DIR"
else
    echo "[1/4] UPV repo already exists, pulling latest..."
    cd "$UPV_REPO_DIR" && git pull && cd "$PROJECT_DIR"
fi

# Check for pre-trained generator in experiments/
echo "[2/4] Looking for pre-trained generator models..."

# The repo provides trained generators in experiments/main_experiments/
# and experiments/robustness/generator_model/
FOUND_GENERATOR=false

# Check main experiments (for OPT-1.3B with top-k sampling)
for dir in \
    "$UPV_REPO_DIR/experiments/main_experiments" \
    "$UPV_REPO_DIR/experiments/robustness/generator_model" \
    "$UPV_REPO_DIR/model"; do
    
    if [ -d "$dir" ]; then
        # Look for sub_net.pt (generator model)
        GENERATOR=$(find "$dir" -name "sub_net.pt" -o -name "generator_model*.pt" 2>/dev/null | head -1)
        if [ -n "$GENERATOR" ]; then
            echo "  Found generator: $GENERATOR"
            cp "$GENERATOR" "$UPV_MODEL_DIR/generator_model_b16_p1.pt"
            FOUND_GENERATOR=true
            break
        fi
    fi
done

if [ "$FOUND_GENERATOR" = false ]; then
    echo "  ⚠ No pre-trained generator found. Will train from scratch (see below)."
fi

# Check for pre-trained detector
echo "[3/4] Looking for pre-trained detector models..."
FOUND_DETECTOR=false

for dir in \
    "$UPV_REPO_DIR/experiments/main_experiments" \
    "$UPV_REPO_DIR/experiments/detector_model" \
    "$UPV_REPO_DIR/model"; do
    
    if [ -d "$dir" ]; then
        DETECTOR=$(find "$dir" -name "detector_model*.pt" -o -name "*detector*.pt" 2>/dev/null | head -1)
        if [ -n "$DETECTOR" ]; then
            echo "  Found detector: $DETECTOR"
            cp "$DETECTOR" "$UPV_MODEL_DIR/detector_model_b16_p1_z4.pt"
            FOUND_DETECTOR=true
            break
        fi
    fi
done

if [ "$FOUND_DETECTOR" = false ]; then
    echo "  ⚠ No pre-trained detector found. Will train from scratch (see below)."
fi

# ─── Summary ───
echo ""
echo "============================================================"
echo "Results:"
echo "============================================================"
ls -la "$UPV_MODEL_DIR/" 2>/dev/null || echo "  (empty)"

if [ "$FOUND_GENERATOR" = true ] && [ "$FOUND_DETECTOR" = true ]; then
    echo ""
    echo "✓ Both models found! You can run the GAN pipeline now:"
    echo "  python main.py --stage pretrain_attacker"
    exit 0
fi

# ─── Option 2: Train from scratch ───
echo ""
echo "============================================================"
echo "Training UPV models from scratch..."
echo "============================================================"
echo ""
echo "This requires 4 steps. Running now..."
echo ""

cd "$UPV_REPO_DIR"

# Install requirements if needed
pip install -r requirements.txt 2>/dev/null || true

# ── Step 1: Generate training data for generator ──
echo "[Step 1/4] Generating training data for generator network..."
echo "  (This does NOT need a GPU or LLM — just random token sequences)"

python generate_data.py \
    --bit_number 16 \
    --window_size 1 \
    --sample_number 2000 \
    --output_file ./train_generator_data/train_generator_data.jsonl

echo "  ✓ Training data generated"

# ── Step 2: Train generator network (sub_net.pt) ──
echo "[Step 2/4] Training generator network (UPVSubNet)..."
echo "  (Small network ~43K params, trains in minutes on CPU)"

python model_key.py \
    --data_dir ./train_generator_data/train_generator_data.jsonl \
    --bit_number 16 \
    --model_dir ./model/ \
    --window_size 1 \
    --layers 5

echo "  ✓ Generator trained → ./model/sub_net.pt"

# Copy generator to project
cp ./model/sub_net.pt "$UPV_MODEL_DIR/generator_model_b16_p1.pt"
echo "  ✓ Copied to $UPV_MODEL_DIR/generator_model_b16_p1.pt"

# ── Step 3: Generate watermarked text data ──
echo "[Step 3/4] Generating watermarked text data using OPT-1.3B..."
echo "  (This NEEDS GPU — generates 10K watermarked + unwatermarked samples)"

python watermark_model.py \
    --bit_number 16 \
    --train_num_samples 10000 \
    --dataset_name c4 \
    --llm_name facebook/opt-1.3b \
    --output_dir ./data \
    --model_dir ./model/ \
    --window_size 1 \
    --layers 5 \
    --use_sampling True \
    --sampling_temp 0.7 \
    --n_beams 0 \
    --max_new_tokens 200 \
    --delta 2.0

echo "  ✓ Watermarked text data generated"

# ── Step 4: Train detector network ──
echo "[Step 4/4] Training detector network..."
echo "  (Uses shared embedding from generator, trains on watermarked/natural text)"

python detector.py \
    --llm_name facebook/opt-1.3b \
    --bit 16 \
    --window_size 1 \
    --input ./data \
    --model_file ./model/sub_net.pt \
    --output_model_dir ./model/ \
    --layers 5 \
    --z_value 4

echo "  ✓ Detector trained"

# Copy detector to project
DETECTOR_FILE=$(find ./model/ -name "detector_model*.pt" | head -1)
if [ -n "$DETECTOR_FILE" ]; then
    cp "$DETECTOR_FILE" "$UPV_MODEL_DIR/detector_model_b16_p1_z4.pt"
    echo "  ✓ Copied to $UPV_MODEL_DIR/detector_model_b16_p1_z4.pt"
fi

cd "$PROJECT_DIR"

echo ""
echo "============================================================"
echo "UPV Setup Complete!"
echo "============================================================"
echo ""
ls -la "$UPV_MODEL_DIR/"
echo ""
echo "Now run: python main.py --stage pretrain_attacker"
