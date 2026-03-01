# Migration Guide: UPV → KGW-SelfHash

## Tại sao phải chuyển

Test chứng minh Static Spoofer **hoàn toàn không hoạt động** với UPV:

```
SFT data (Spoofer):  green ratio = 46% (= random), z-score = -1.0
Oracle watermarked:   green ratio = 75%, z-score = +8.0
```

**Nguyên nhân gốc:** UPV dùng neural network để quyết định green/red — mapping từ token ID sang green/red phụ thuộc vào learned weights, không có pattern frequency-based nào cho Spoofer khai thác. Phương pháp watermark-stealing (frequency ratio) được thiết kế cho hash-based watermarks.

**KGW-SelfHash** dùng hash (PRF) → cùng context luôn cho cùng green list → frequency stealing bắt được pattern → Spoofer hoạt động (paper gốc: 80%+ spoofing rate).

## Kiến trúc mới

```
┌─────────────────────────────────────────────────────────┐
│  KGW Oracle (frozen)                                     │
│  ├── OPT-1.3B + KGWLogitsProcessor (γ=0.25, δ=2.0)     │
│  └── Generate real watermarked text                      │
├─────────────────────────────────────────────────────────┤
│  KGW Discriminator (z-score, NON-trainable)              │
│  ├── hash(context, key) → green list                     │
│  ├── Count green tokens → z-score                        │
│  └── reward = sigmoid((z - 2.0) / 2.0)                  │
├─────────────────────────────────────────────────────────┤
│  Attacker (OPT-1.3B + LoRA, TRAINABLE)                  │
│  ├── Step 1: Watermark Stealing → Static Spoofer        │
│  ├── Step 2: SFT on Spoofer output (warm-start)         │
│  └── Step 3: Adversarial RL (MC Search + REINFORCE)     │
└─────────────────────────────────────────────────────────┘
```

### Thay đổi quan trọng: D không trainable

Với KGW, D = z-score computation (pure math, deterministic). Không cần train D → không có vấn đề "D quá mạnh" → GAN training ổn định hơn nhiều.

Reward signal luôn smooth nhờ sigmoid mapping:
- z=0 (random text): reward ≈ 0.27
- z=2 (weak watermark): reward = 0.50
- z=4 (threshold): reward ≈ 0.73
- z=8 (strong watermark): reward ≈ 0.95

## Files cần tạo/thay

### Files MỚI (copy vào repo)

| File | Mô tả |
|------|--------|
| `watermark/kgw_watermark.py` | KGW core: Oracle, Detector, LogitsProcessor |
| `watermark/kgw_discriminator.py` | Z-score based discriminator cho GAN |
| `watermark/__init__.py` | Package init |
| `config/gan_config_kgw.yaml` | Config mới |

### Files CẦN SỬA

#### 1. `main.py` — Thay UPV imports bằng KGW

```python
# CŨ:
from models.upv_discriminator import UPVDiscriminatorWrapper
from data.data_generator import UPVOracle

# MỚI:
from watermark.kgw_watermark import KGWOracle
from watermark.kgw_discriminator import KGWDiscriminator
```

Trong `run_adversarial()`:
```python
# CŨ:
oracle = UPVOracle(model_name=config.llm_name, ...)
discriminator = UPVDiscriminatorWrapper(bit_number=16, ...)

# MỚI:
oracle = KGWOracle(
    model_name=config.llm_name,
    device=device,
    gamma=config.wm_gamma,
    delta=config.wm_delta,
    context_width=config.wm_context_width,
    hash_key=config.wm_hash_key,
)
discriminator = KGWDiscriminator(
    vocab_size=oracle.vocab_size,
    gamma=config.wm_gamma,
    delta=config.wm_delta,
    context_width=config.wm_context_width,
    hash_key=config.wm_hash_key,
    z_center=config.disc_z_center,
    temperature=config.disc_temperature,
    device=device,
)
```

#### 2. `models/attacker.py` — WatermarkLearner prevctx_width=3

Đã hỗ trợ configurable. Chỉ cần config truyền `prevctx_width=3` (match KGW h=3).

**QUAN TRỌNG:** Với KGW hash-based, frequency stealing SẼ hoạt động:
- Cùng context → cùng green list (deterministic)
- Token xuất hiện thường xuyên sau context X trong WM text → likely green
- Ratio WM/base frequency > threshold → boost

#### 3. `training/adversarial_loop.py` — Bỏ D update

```python
# CŨ (trong train()):
for _ in range(self.config.adv_d_steps):
    d_metrics = self.update_discriminator(...)

# MỚI:
# D is z-score based — no training needed
d_metrics = {'d_loss': 0.0, 'd_reward_real': 0.0, 'd_reward_fake': 0.0}
# Optionally compute metrics for logging
if epoch % self.config.adv_eval_every == 0:
    d_metrics = self._compute_d_metrics(mc_result, real_padded, real_lengths)
```

#### 4. `utils/helpers.py` — Thêm KGW config fields vào GANConfig

```python
# Thêm vào dataclass:
wm_gamma: float
wm_delta: float
wm_context_width: int
wm_hash_key: int
disc_z_center: float
disc_temperature: float

# Thêm vào load_config():
wm_gamma=cfg['watermark']['gamma'],
wm_delta=cfg['watermark']['delta'],
wm_context_width=cfg['watermark']['context_width'],
wm_hash_key=cfg['watermark']['hash_key'],
disc_z_center=cfg['discriminator']['z_center'],
disc_temperature=cfg['discriminator']['temperature'],
```

#### 5. `evaluation/metrics.py` — Dùng KGW detector

```python
# CŨ:
def compute_upv_detection_rate(oracle, texts):
    result = oracle.detect_watermark(text)
    score = result.get('score', 0)

# MỚI (interface giống, chỉ key khác):
def compute_kgw_detection_rate(oracle, texts):
    result = oracle.detect_watermark(text)
    z_score = result.get('z_score', 0)
    is_wm = result.get('is_watermarked', False)
```

## Thứ tự thực hiện

```bash
# 1. Tạo thư mục watermark/
mkdir -p watermark/
cp kgw_watermark.py watermark/
cp kgw_discriminator.py watermark/
echo "from .kgw_watermark import *" > watermark/__init__.py
echo "from .kgw_discriminator import *" >> watermark/__init__.py

# 2. Copy config mới
cp config/gan_config_kgw.yaml config/gan_config.yaml

# 3. Sửa main.py, adversarial_loop.py, helpers.py, metrics.py
# (theo hướng dẫn ở trên)

# 4. Xóa cache cũ
rm -rf checkpoints/

# 5. Test KGW watermark trước
python -c "
from watermark.kgw_watermark import KGWOracle
oracle = KGWOracle('facebook/opt-1.3b', device='cuda')
wm, _ = oracle.generate_watermarked(['The research suggests that'])
result = oracle.detect_watermark(wm[0])
print(f'z-score: {result[\"z_score\"]:.2f}, green: {result[\"green_fraction\"]:.2%}')
"

# 6. Train pipeline
python main.py --stage pretrain_attacker
python main.py --stage adversarial     # D không cần pretrain nữa!
python main.py --stage evaluate
```

## Kỳ vọng

| Metric | UPV (trước) | KGW (kỳ vọng) |
|--------|-------------|----------------|
| Spoofer green ratio | 46% (random) | **65-75%** |
| SFT z-score | -1.0 | **3-6** |
| Spoofing rate (after GAN) | 0% | **40-70%** |
| Training stability | Mode collapse | **Stable** (smooth reward) |
| D too strong? | Luôn xảy ra | **Không** (z-score reward smooth) |

## Ưu điểm của thiết kế mới

1. **D không trainable** → không có D/G imbalance, arms race ổn định
2. **Reward smooth** → REINFORCE gradient luôn có hướng, không vanish
3. **Spoofer hoạt động** → SFT warm-start mạnh → G bắt đầu từ vị trí tốt
4. **Z-score interpretable** → dễ debug, biết chính xác watermark strength
5. **Paper contribution rõ ràng** → "GAN adversarial training as defense against watermark stealing attack on KGW"
