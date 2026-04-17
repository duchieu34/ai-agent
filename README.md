# Reply AI Agent Challenge 2026 - Multi-Agent Fraud Detection

Pipeline phát hiện giao dịch gian lận sử dụng multi-agent LLM orchestration.

## 1. Tổng quan kiến trúc

```
Planner ─→ Extractor ─→ Scorer ─→ Critic ─→ Finalize
   │                                            ↑
   └──── Fast path: Planner → Decider ──────────┘
```

- **Planner**: phân tích dataset, xác định chiến lược & tín hiệu ưu tiên.
- **Extractor**: chọn lọc transaction ID đáng ngờ từ candidate pool.
- **Scorer**: xếp hạng và cho điểm từng ID, loại bỏ false positive.
- **Critic**: review & veto cuối cùng, giữ lại ID có confidence cao.
- **Finalize**: quality-gated backfill, precision filter, submission firewall.

Challenge mode luôn force critic (không skip) để tăng precision.

### Feature engineering (risk_hint)

Adapter `challenge.py` tính `risk_hint` cho mỗi transaction từ nhiều component:
`c_amt`, `c_sender_z`, `c_recipient_z`, `c_pair_rare`, `c_method_rare`, `c_type_rare`, `c_hour_rare`, `c_night`, `c_rapid`, `c_desc_kw`, `c_struct`, `c_balance`, `c_mail`, `c_pair_high`, `c_benign`.

Backfill chỉ lấy ID có risk_hint >= threshold (1.0 high, 0.6 medium).

## 2. Cấu trúc thư mục

```
├── run_pipeline.py          # Entry point
├── src/mirrorlife_agent/
│   ├── cli.py               # CLI argument parser
│   ├── config.py            # Settings từ env
│   ├── orchestrator.py      # Multi-agent orchestration core
│   ├── adapters/            # Adapter theo mode (sandbox/challenge)
│   ├── agents/              # Planner, Extractor, Scorer, Critic, Decider
│   ├── openrouter_client.py # LLM client (OpenRouter/OpenAI)
│   ├── budget_guard.py      # Token & cost limiter
│   ├── submission_guard.py  # Output validation firewall
│   ├── replay_logger.py     # JSON replay log
│   └── tracing.py           # Langfuse integration
├── outputs/                 # Kết quả output (.txt)
├── replays/                 # JSON replay log cho debug
├── scripts/                 # Utility scripts (sweep, benchmark)
├── The Truman Show - train/ # Dataset Lev1
├── Brave New World - train/ # Dataset Lev2
├── Deus Ex - train/         # Dataset Lev3 (có audio/)
├── Deus Ex - validation/    # Dataset Lev3 validation
└── .env                     # Cấu hình runtime
```

## 3. Cài đặt

```powershell
# Tạo venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt

# Chuẩn bị env
copy .env.example .env
# Điền API key vào .env
```

## 4. Cách chạy pipeline

### Cú pháp chung

```powershell
& ".venv\Scripts\python.exe" run_pipeline.py run `
    --mode {sandbox,challenge} `
    --phase {training,evaluation} `
    --dataset-key <KEY> `
    --dataset "<DATASET_FOLDER>" `
    --output "outputs/<FILE>.txt" `
    --max-output-ids <N>
```

| Tham số | Mô tả |
|---------|-------|
| `--mode` | `sandbox` (Citizen ID) hoặc `challenge` (Transaction ID) |
| `--phase` | `training` (có ground truth) hoặc `evaluation` (submit) |
| `--dataset-key` | Tên định danh cho lần chạy (hiển thị trong replay) |
| `--dataset` | Đường dẫn thư mục dataset |
| `--output` | File output kết quả |
| `--max-output-ids` | Giới hạn số ID tối đa trong output |

### 4.1 Sandbox (dữ liệu public)

```powershell
# Lev 1
& ".venv\Scripts\python.exe" run_pipeline.py run --mode sandbox --phase training --dataset-key public_lev_1 --dataset public_lev_1.zip --output outputs/public_lev_1.txt

# Lev 2
& ".venv\Scripts\python.exe" run_pipeline.py run --mode sandbox --phase training --dataset-key public_lev_2 --dataset public_lev_2.zip --output outputs/public_lev_2.txt

# Lev 3
& ".venv\Scripts\python.exe" run_pipeline.py run --mode sandbox --phase training --dataset-key public_lev_3 --dataset public_lev_3.zip --output outputs/public_lev_3.txt
```

### 4.2 Challenge - Training

```powershell
# Lev 1 - The Truman Show
& ".venv\Scripts\python.exe" run_pipeline.py run --mode challenge --phase training --dataset-key lev1_the_truman --dataset "The Truman Show - train" --output "outputs/lev1_the_truman.txt" --max-output-ids 80

# Lev 2 - Brave New World
& ".venv\Scripts\python.exe" run_pipeline.py run --mode challenge --phase training --dataset-key lev2_brave_new --dataset "Brave New World - train" --output "outputs/lev2_brave_new.txt" --max-output-ids 80

# Lev 3 - Deus Ex
& ".venv\Scripts\python.exe" run_pipeline.py run --mode challenge --phase training --dataset-key lev3_deus_ex --dataset "Deus Ex - train" --output "outputs/lev3_deus_ex.txt" --max-output-ids 80
```

### 4.3 Challenge - Evaluation (submit)

```powershell
# Lev 3 - Deus Ex validation
& ".venv\Scripts\python.exe" run_pipeline.py run --mode challenge --phase evaluation --dataset-key lev3_deus_ex_val --dataset "Deus Ex - validation" --output "outputs/lev3_deus_ex_val.txt" --max-output-ids 80
```

### 4.4 Kiểm tra trạng thái submission

```powershell
& ".venv\Scripts\python.exe" run_pipeline.py status
```

## 5. Biến môi trường quan trọng

Đầy đủ trong `.env.example`. Nhóm chính:

### 5.1 Provider & model

| Biến | Mô tả |
|------|-------|
| `LLM_PROVIDER` | `openrouter` hoặc `openai` |
| `OPENROUTER_API_KEY` | API key OpenRouter |
| `OPENROUTER_MODEL` | Model mặc định (vd: `google/gemini-2.0-flash-001`) |
| `OPENAI_API_KEY` | API key OpenAI |
| `OPENAI_MODEL` | Model mặc định (vd: `gpt-4o-mini`) |

### 5.2 Model theo vai trò agent (tùy chọn)

`LLM_MODEL_PLANNER`, `LLM_MODEL_DECIDER`, `LLM_MODEL_EXTRACTOR`, `LLM_MODEL_SCORER`, `LLM_MODEL_CRITIC`

Nếu để trống → dùng model mặc định theo provider.

### 5.3 Điều khiển pipeline

| Biến | Default | Mô tả |
|------|---------|-------|
| `DECISION_PROFILE` | `precision_first` | Chiến lược finalize |
| `BUDGET_PROFILE` | `auto` | `auto` / `low` / `high` |
| `FORCE_FULL_CHAIN` | `false` | Luôn chạy full chain |
| `ADAPTIVE_CHAIN_ENABLED` | `true` | Cho phép adaptive routing |
| `FAST_PATH_MIN_CONFIDENCE` | `0.80` | Ngưỡng confidence cho fast path |
| `CRITIC_MIN_CONFIDENCE` | `0.65` | Ngưỡng để skip critic |
| `SUBMISSION_MAX_FLAGGED_RATIO` | `0.60` | Tỷ lệ flag tối đa |
| `CHALLENGE_MIN_FLAGGED_RATIO` | `0.22` | Tỷ lệ flag tối thiểu (challenge) |
| `CHALLENGE_FALLBACK_MIN_FLAGGED_RATIO` | `0.10` | Tỷ lệ flag tối thiểu khi fallback |
| `RISK_ELBOW_CAP_ENABLED` | `true` | Bật cắt elbow trên risk_hint |

### 5.4 Tracing (Langfuse)

| Biến | Mô tả |
|------|-------|
| `LANGFUSE_PUBLIC_KEY` | Public key |
| `LANGFUSE_SECRET_KEY` | Secret key |
| `LANGFUSE_HOST` | `https://challenges.reply.com/langfuse` |
| `TEAM_NAME` | Tên team (vd: `VNBrain2`) |
| `ENFORCE_LANGFUSE` | `true` để bắt buộc tracing |

### 5.5 Replay & debug

| Biến | Default | Mô tả |
|------|---------|-------|
| `REPLAY_LOG_ENABLED` | `true` | Ghi replay JSON |
| `REPLAY_LOG_DIR` | `replays` | Thư mục lưu replay |

## 6. Output contract

- Plain text ASCII, mỗi dòng 1 ID, không header.
- `sandbox` → Citizen ID
- `challenge` → Transaction ID

## 7. Debug & replay

Mỗi lần chạy sinh file JSON trong `replays/` chứa:

- `final_ids`: danh sách ID output cuối
- `chain_path`: luồng chạy (`full-with-critic` / `fast`)
- `finalize_debug`: chi tiết source votes, backfill, precision filter
- `tool_features_preview`: feature engineering cho từng transaction
- `risk_component_summary`: thống kê component risk

Đọc nhanh replay:

```powershell
& ".venv\Scripts\python.exe" -c "
import json, glob
f = sorted(glob.glob('replays/*.json'))[-1]
r = json.load(open(f))
print('key:', r['dataset_key'])
print('chain:', r['chain_path'])
print('final_ids:', len(r['final_ids']))
print('flagged_ratio:', r['firewall_report']['flagged_ratio'])
"
```

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
|-----|-------------|
| `--phase validation` không hợp lệ | Dùng `--phase evaluation` |
| Thiếu API key | Kiểm tra `.env` đã điền key đúng provider |
| Rate limit | Giảm tải hoặc chuyển model, kiểm tra retry config |
| Output bị từ chối | Kiểm tra output contract: ASCII, 1 ID/dòng, không header |
| Scorer fallback | Bình thường - fallback lấy subset từ extractor, precision filter sẽ lọc thêm |
