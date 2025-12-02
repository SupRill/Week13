# Stockout Prediction (Logistic Regression / SVM) — Streamlit App

Aplikasi Streamlit untuk memprediksi apakah sebuah barang akan **stockout** (True/False) berdasarkan level inventaris dan lead time.

## File
- `app.py` — Streamlit app utama
- `requirements.txt` — dependensi

## Format dataset (CSV)
Minimal kolom:
- `inventory_level` (numerik)
- `lead_time` (numerik)
- `stockout_indicator` (target, biner: True/False, 1/0, Ya/Tidak)

Anda dapat menggunakan nama kolom target lain; saat upload aplikasi akan meminta Anda memilih kolom target jika tidak terdeteksi.

## Cara pakai lokal
1. Buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate     # Windows
