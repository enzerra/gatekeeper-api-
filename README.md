# SIKARTA - AI Gatekeeper API (Computer Vision & OCR)

Ini adalah layanan mandiri (*Microservice*) berbasis Kecerdasan Buatan (AI) yang bertanggung jawab penuh dalam memvalidasi keaslian gambar kwitansi dan mengekstrak teks nominal transaksi (OCR) secara otomatis.

## 🚀 Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python 3.10+
- **Framework API:** FastAPI
- **Image Classification (Fraud Detection):** TensorFlow / Keras (MobileNetV2)
- **OCR & NLP Engine:** Google Generative AI (Gemini 1.5)
- **Deployment:** Hugging Face Spaces (Docker Environment)

## ⚙️ Persyaratan Sistem
- Python 3.10 atau yang lebih baru.

## 🛠️ Cara Menjalankan di Komputer Lokal

1. **Install Dependensi**
   Sangat disarankan untuk menggunakan *Virtual Environment* (venv). Jalankan perintah berikut di terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Konfigurasi API Key**
   Buat file `.env` (atau salin dari `.env.example`) dan isi dengan Google Gemini API Key Anda:
   ```env
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

3. **Jalankan Server AI**
   Jalankan Uvicorn server untuk mengaktifkan FastAPI:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
   Server AI akan berjalan di `http://127.0.0.1:8000`.

## 📡 Dokumentasi API (Swagger UI)
Setelah server berjalan, Anda dapat menguji endpoint pemindaian AI secara langsung melalui UI interaktif bawaan FastAPI.
Buka browser dan akses: `http://127.0.0.1:8000/docs`

## ☁️ Deployment ke Hugging Face
Saat melakukan deploy ke Hugging Face Spaces, pastikan Anda mengisi *Secret Token* `GEMINI_API_KEY` pada pengaturan proyek di web Hugging Face agar sistem tidak mengalami *Error 500 Unauthorized*.
