1. **Rationale:** Protokol ini direvisi total untuk mendukung bootstrap lingkungan dari nol, batasan RAM 4GB yang ketat, dan arsitektur CPU-only tanpa dependensi awal.

2. **The Code:**

```markdown
# autoresearch-cpu-slm

Eksperimen otonom untuk membangun Small Language Model (SLM) chat pada hardware terbatas (CPU Single Core, 4GB RAM) dari lingkungan kosong.

## Setup & Bootstrap

Karena folder kosong, agent HARUS menginisialisasi lingkungan sebelum eksperimen:

1. **Inisialisasi Environment**:
   - Install Python 3.10+ jika belum ada.
   - Install `uv` atau `pip`.
   - Buat virtual environment: `python -m venv venv`.
   - Buat `pyproject.toml` dengan dependensi minimal: `torch` (CPU only), `transformers`, `datasets`, `accelerate`.
   - **Penting**: Pastikan torch versi CPU (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) untuk hemat RAM.
2. **Buat Struktur File**:
   - `prepare.py`: Script untuk download dataset kecil (e.g., TinyStories, Alpaca subset) dan tokenizer.
   - `train.py`: Script training loop yang ramah CPU.
   - `inference.py`: Script untuk testing chat setelah training.
   - `results.tsv`: Log eksperimen.
3. **Verifikasi RAM**: Pastikan penggunaan RAM awal < 1GB sebelum training.
4. **Konfirmasi**: Laporkan kesiapan lingkungan ke user hanya jika setup gagal total.

## Eksperimentasi

Script berjalan pada **CPU Single Core**. Budget waktu **60 menit per run** (CPU lebih lambat). Jalankan: `python train.py`.

**Yang BOLEH dilakukan:**
- Modifikasi `train.py` & `prepare.py` (karena anda yang membuatnya).
- Arsitektur Model: Wajib **< 10 Million Parameters** (agar muat di 4GB RAM saat training).
- Gunakan `gradient_accumulation` untuk simulasi batch size besar tanpa meledakkan RAM.
- Gunakan `dtype=torch.float32` atau `float16` (jika CPU support AVX512), hindari mixed precision jika tidak stabil.

**Yang TIDAK BOLEH dilakukan:**
- Menggunakan library yang membutuhkan CUDA/GPU.
- Membiarkan penggunaan RAM melebihi 3.5GB (sisakan buffer OS).
- Download dataset > 500MB (hemat storage & RAM).

**Tujuan**: Model bisa melakukan chat sederhana (bahasa Inggris/Indonesia dasar).
**Constraint Utama**: RAM 4GB. OOM = Crash.
**Kriteria Kesederhanaan**: Model lebih kecil lebih baik. Prioritaskan arsitektur efisien (LSTM, GRU, atau Transformer 2 layer).

**Run Pertama**: Buat baseline model terkecil yang bisa jalan (misal: 1 layer, 1M params).

## Output Format

Script mencetak summary:

```
---
val_loss:         2.5000
training_seconds: 3600.0
peak_ram_mb:      3200.5
num_params_M:     5.4
```

Ekstrak metrik kunci:
```
grep "^val_loss:\|^peak_ram_mb:" run.log
```

## Logging Results

Catat ke `results.tsv` (tab-separated).

Header:
```
commit	val_loss	memory_gb	status	description
```

1. git commit hash (7 chars)
2. val_loss (6 desimal) — 0.000000 jika crash
3. peak memory GB (peak_ram_mb / 1024)
4. status: `keep`, `discard`, `crash`
5. deskripsi singkat

## The Experiment Loop

LOOP FOREVER (dengan safety check ketat):

1. Cek status git.
2. Modifikasi kode (`train.py`, `prepare.py`, `pyproject.toml` jika perlu).
3. `git commit`.
4. Jalankan: `python train.py > run.log 2>&1`.
5. Monitor RAM secara eksternal jika memungkinkan (task manager/top).
6. Baca hasil: `grep "^val_loss:\|^peak_ram_mb:" run.log`.
7. Jika crash (OOM/Error): Cek log. Perbaiki bug. Jika ide terlalu berat, discard.
8. Catat di `results.tsv`.
9. Jika `val_loss` turun DAN `peak_ram_mb` < 3500: **Keep**.
10. Jika `val_loss` naik atau RAM > 3.5GB: **Discard** (git reset).

**Timeout**: Jika run > 90 menit, kill dan anggap gagal.
**Safety**: Jika sistem hang/unresponsive, hentikan loop.
**Autonomy**: Lanjutkan tanpa bertanya kecuali error sistem kritis.

**Catatan Realistis**:
- Training dari nol pada 4GB RAM sangat terbatas.
- Jika training dari scratch gagal menghasilkan model chat yang koheren setelah 5 iterasi, beralih strategi: Download pre-trained tiny model (e.g., Qwen-0.5B quantized) dan lakukan fine-tuning ringan (LoRA) jika RAM memungkinkan.
- Fokus pada kualitas data daripada ukuran model.
```
