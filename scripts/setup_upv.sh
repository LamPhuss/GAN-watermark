#!/bin/bash
# ============================================================
# setup_upv.sh
# Download pre-trained UPV weights from official repo
# OR train from scratch
#
# FIX 1: All --window_size changed from 1 to 3
#        Output filenames updated to reflect w3
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UPV_MODEL_DIR="$PROJECT_DIR/upv/model"
UPV_REPO_DIR="$PROJECT_DIR/unforgeable_watermark"

# ── Configuration ──
WINDOW_SIZE=3    # FIX 1: was 1, now 3
BIT_NUMBER=16
LAYERS=5
Z_VALUE=4
DELTA=2.0
LLM_NAME="facebook/opt-1.3b"

mkdir -p "$UPV_MODEL_DIR"

echo "============================================================"
echo "UPV Model Setup"
echo "  window_size=$WINDOW_SIZE, bit=$BIT_NUMBER, layers=$LAYERS"
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

FOUND_GENERATOR=false

for dir in \
    "$UPV_REPO_DIR/experiments/robustness/generator_model/opt-1.3b/window_size_${WINDOW_SIZE}" \
    "$UPV_REPO_DIR/experiments/main_experiments/generator_model" \
    "$UPV_REPO_DIR/model"; do

    if [ -d "$dir" ]; then
        GENERATOR=$(find "$dir" -name "sub_net.pt" -o -name "generator_model*.pt" 2>/dev/null | head -1)
        COMBINE=$(find "$dir" -name "combine_model.pt" 2>/dev/null | head -1)
        if [ -n "$GENERATOR" ]; then
            echo "  Found generator: $GENERATOR"
            cp "$GENERATOR" "$UPV_MODEL_DIR/generator_model_b${BIT_NUMBER}_w${WINDOW_SIZE}.pt"
            if [ -n "$COMBINE" ]; then
                cp "$COMBINE" "$UPV_MODEL_DIR/combine_model.pt"
            fi
            FOUND_GENERATOR=true
            break
        fi
    fi
done

if [ "$FOUND_GENERATOR" = false ]; then
    echo "  ⚠ No pre-trained generator found. Will train from scratch."
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
            cp "$DETECTOR" "$UPV_MODEL_DIR/detector_model_b${BIT_NUMBER}_w${WINDOW_SIZE}_z${Z_VALUE}.pt"
            FOUND_DETECTOR=true
            break
        fi
    fi
done

if [ "$FOUND_DETECTOR" = false ]; then
    echo "  ⚠ No pre-trained detector found. Will train from scratch."
fi

# If both found, done
if [ "$FOUND_GENERATOR" = true ] && [ "$FOUND_DETECTOR" = true ]; then
    echo ""
    echo "============================================================"
    echo "UPV Setup Complete! (pre-trained weights)"
    echo "  Generator: $UPV_MODEL_DIR/generator_model_b${BIT_NUMBER}_w${WINDOW_SIZE}.pt"
    echo "  Detector:  $UPV_MODEL_DIR/detector_model_b${BIT_NUMBER}_w${WINDOW_SIZE}_z${Z_VALUE}.pt"
    echo ""
    echo "You can run the GAN pipeline now:"
    echo "  python main.py --stage pretrain_attacker"
    exit 0
fi

# ─── Option 2: Train from scratch ───
echo ""
echo "============================================================"
echo "Training UPV models from scratch..."
echo "  window_size=$WINDOW_SIZE, layers=$LAYERS"
echo "============================================================"
echo ""

cd "$UPV_REPO_DIR"

pip install -r requirements.txt 2>/dev/null || true

# ── Step 1: Generate training data for generator ──
echo "[Step 1/4] Generating training data for generator network..."

python generate_data.py \
    --bit_number $BIT_NUMBER \
    --window_size $WINDOW_SIZE \
    --sample_number 2000 \
    --output_file ./train_generator_data/train_generator_data.jsonl

echo "  ✓ Training data generated"

# ── Step 2: Train generator network (sub_net.pt) ──
echo "[Step 2/4] Training generator network (300 epochs)..."

python model_key.py \
    --data_dir ./train_generator_data/train_generator_data.jsonl \
    --bit_number $BIT_NUMBER \
    --model_dir ./model/ \
    --window_size $WINDOW_SIZE \
    --layers $LAYERS

echo "  ✓ Generator trained → ./model/sub_net.pt"

# Copy generator to project
cp ./model/sub_net.pt "$UPV_MODEL_DIR/generator_model_b${BIT_NUMBER}_w${WINDOW_SIZE}.pt"
cp ./model/combine_model.pt "$UPV_MODEL_DIR/combine_model.pt"
echo "  ✓ Copied to $UPV_MODEL_DIR/"

# ── Step 3: Generate watermarked text data ──
echo "[Step 3/4] Generating watermarked text data using ${LLM_NAME}..."

python watermark_model.py \
    --bit_number $BIT_NUMBER \
    --train_num_samples 10000 \
    --dataset_name c4 \
    --llm_name $LLM_NAME \
    --output_dir ./data \
    --model_dir ./model/ \
    --window_size $WINDOW_SIZE \
    --layers $LAYERS \
    --use_sampling True \
    --sampling_temp 0.7 \
    --n_beams 0 \
    --max_new_tokens 200 \
    --delta $DELTA

echo "  ✓ Watermarked text data generated"

# ── Step 4: Train detector network ──
echo "[Step 4/4] Training detector network (80 epochs, lr=0.0005)..."

python detector.py \
    --llm_name $LLM_NAME \
    --bit $BIT_NUMBER \
    --window_size $WINDOW_SIZE \
    --input ./data \
    --model_file ./model/sub_net.pt \
    --output_model_dir ./model/ \
    --layers $LAYERS \
    --z_value $Z_VALUE

echo "  ✓ Detector trained"

# Copy detector to project
DETECTOR_FILE=$(find ./model/ -name "detector_model*.pt" | head -1)
if [ -n "$DETECTOR_FILE" ]; then
    cp "$DETECTOR_FILE" "$UPV_MODEL_DIR/detector_model_b${BIT_NUMBER}_w${WINDOW_SIZE}_z${Z_VALUE}.pt"
    echo "  ✓ Copied to $UPV_MODEL_DIR/"
fi

cd "$PROJECT_DIR"

echo ""
echo "============================================================"
echo "UPV Setup Complete!"
echo "  Generator: $UPV_MODEL_DIR/generator_model_b${BIT_NUMBER}_w${WINDOW_SIZE}.pt"
echo "  Combine:   $UPV_MODEL_DIR/combine_model.pt"
echo "  Detector:  $UPV_MODEL_DIR/detector_model_b${BIT_NUMBER}_w${WINDOW_SIZE}_z${Z_VALUE}.pt"
echo ""
echo "Next steps:"
echo "  python main.py --stage pretrain_attacker"
echo "  python main.py --stage pretrain_detector"
echo "  python main.py --stage adversarial"
echo "============================================================"
