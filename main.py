import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.api_core import exceptions as gexc
from google.api_core.retry import Retry
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Muat variabel dari file .env
load_dotenv()

# Ambil API key secara aman
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GEMINI_API_KEY belum di-set; endpoint Gemini akan gagal.")

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Inisialisasi Aplikasi FastAPI
app = FastAPI(title="AI Gatekeeper & Extractor API - Karang Taruna")

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
allow_credentials = allowed_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model AI Satpam (MobileNetV2)
class CustomDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

model_satpam: Optional[tf.keras.Model] = None
model_load_error = None
try:
    model_satpam = tf.keras.models.load_model(
        "gatekeeper_model.keras",
        custom_objects={'Dense': CustomDense}
    )
    logger.info("Model AI Gatekeeper (Satpam) berhasil dimuat.")
except Exception as exc:
    model_load_error = str(exc)
    logger.error("Gagal memuat model: %s", exc)

@app.get("/")
def home():
    return {"message": "Sistem AI Karang Taruna Aktif!"}

def _read_image(contents: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.") from exc


def _extract_json(text: str) -> Any:
    cleaned = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _downscale_image(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return img.resize(new_size)


def _prepare_gemini_payload(img: Image.Image, max_side: int) -> Any:
    quality = int(os.getenv("GEMINI_JPEG_QUALITY", "75"))
    use_jpeg = os.getenv("GEMINI_USE_JPEG", "1") == "1"
    resized = _downscale_image(img, max_side)
    if not use_jpeg:
        return resized
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return buffer.read()


def _call_gemini(prompt: str, img: Image.Image) -> Any:
    timeout_seconds = int(os.getenv("GEMINI_TIMEOUT", "120"))
    primary_max_side = int(os.getenv("GEMINI_MAX_SIDE", "1600"))
    retry_max_side = int(os.getenv("GEMINI_RETRY_MAX_SIDE", "1000"))
    prepared_img = _prepare_gemini_payload(img, primary_max_side)
    try:
        response = gemini_model.generate_content(
            [prompt, prepared_img],
            request_options={"timeout": timeout_seconds, "retry": Retry(maximum=0)},
        )
        return _extract_json(response.text)
    except (TypeError, ValueError):
        response = gemini_model.generate_content(
            [prompt, _downscale_image(img, primary_max_side)],
            request_options={"timeout": timeout_seconds, "retry": Retry(maximum=0)},
        )
        return _extract_json(response.text)
    except gexc.DeadlineExceeded as exc:
        try:
            smaller_img = _prepare_gemini_payload(img, retry_max_side)
            response = gemini_model.generate_content(
                [prompt, smaller_img],
                request_options={"timeout": timeout_seconds, "retry": Retry(maximum=0)},
            )
            return _extract_json(response.text)
        except gexc.DeadlineExceeded as retry_exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Gemini timeout. Coba lagi atau gunakan gambar dengan resolusi lebih kecil. "
                    "Anda bisa set GEMINI_MAX_SIDE dan GEMINI_TIMEOUT."
                ),
            ) from retry_exc


def _split_left_right(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    overlap = int(os.getenv("SPLIT_OVERLAP_PX", "20"))
    width, height = img.size
    mid = width // 2
    left = img.crop((0, 0, mid + overlap, height))
    right = img.crop((mid - overlap, 0, width, height))
    return left, right


def _require_gemini() -> None:
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY belum di-set.")


def _require_model() -> tf.keras.Model:
    if model_satpam is None:
        raise HTTPException(status_code=503, detail=f"Model klasifikasi belum siap. Error: {model_load_error}")
    return model_satpam


@app.post("/predict")
async def predict_and_extract(file: UploadFile = File(...)):
    try:
        model = _require_model()
        contents = await file.read()
        img_pil = _read_image(contents)

        img_resized = img_pil.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        score = float(prediction[0][0])

        if score > 0.1: # Turunkan threshold agar gambar lebih sering diteruskan ke Gemini untuk dicek silang
            _require_gemini()
            prompt = """
            Anda adalah asisten data keuangan cerdas berlapis ganda.
            TUGAS PERTAMA: Verifikasi apakah gambar ini BENAR-BENAR sebuah kwitansi, struk, nota, faktur, atau bukti pembayaran yang sah.
            Jika gambar ini jelas-jelas BUKAN bukti pembayaran (misalnya foto wajah, hewan, pemandangan, screenshot game, dll), Anda WAJIB mengembalikan JSON persis seperti ini HANYA:
            {
              "is_valid": false
            }

            TUGAS KEDUA: Jika gambar ini ADALAH bukti pembayaran yang sah, ekstrak informasi berikut:
            1. Nama Toko atau Instansi
            2. Alamat Toko (jika ada, tulis selengkapnya. Jika tidak ada tulis "Tidak tersedia")
            3. Tanggal Transaksi
            4. Daftar Item Belanja (ekstrak setiap baris barang yang dibeli: nama barang, jumlah/qty, harga satuan, dan subtotal per barang)
            5. Total Bayar keseluruhan (berupa angka murni tanpa titik/koma pemisah ribuan)
            6. Kategori yang Disarankan (suggested_category). Pilih SATU HANYA dari daftar ini: "Konsumsi", "Operasional", "Transportasi", "Perlengkapan", "Lainnya".

            Format balasan Anda HARUS dalam bentuk JSON murni dengan struktur persis seperti ini:
            {
              "is_valid": true,
              "nama_toko": "...",
              "alamat": "...",
              "tanggal": "...",
              "suggested_category": "...",
              "daftar_item": [
                {
                  "nama_barang": "...",
                  "qty": 0,
                  "harga_satuan": 0,
                  "subtotal": 0
                }
              ],
              "total_bayar": 0
            }
            Jangan tambahkan teks pembuka, penutup, atau tanda markdown seperti ```json. Berikan JSON mentah saja.
            """

            data_kwitansi = _call_gemini(prompt, img_pil)

            # Jika Gemini memvonis ini BUKAN struk
            if not data_kwitansi.get("is_valid", True):
                return {
                    "status": "success",
                    "filename": file.filename,
                    "klasifikasi": "Bukan_Bukti",
                    "confidence": "99.9%",
                    "message": "Gambar tidak valid (Ditolak oleh Gemini AI lapis kedua). Ekstraksi data dibatalkan.",
                }

            return {
                "status": "success",
                "filename": file.filename,
                "klasifikasi": "Kwitansi_Valid",
                "confidence": f"{round(score * 100, 2)}%",
                "data_ekstraksi": data_kwitansi,
            }

        return {
            "status": "success",
            "filename": file.filename,
            "klasifikasi": "Bukan_Bukti",
            "confidence": f"{round((1 - score) * 100, 2)}%",
            "message": "Gambar tidak valid. Ekstraksi data dibatalkan.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gagal memproses prediksi")
        raise HTTPException(status_code=500, detail=str(exc))
    

# ==========================================
# FITUR 1: SCAN BUKU CATATAN FISIK -> EXCEL
# ==========================================
# ==========================================
# FITUR 1A: SCAN PER HALAMAN (KEMBALIKAN JSON)
# ==========================================
@app.post("/scan-halaman")
async def scan_halaman_buku(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_pil = _read_image(contents)
        
        # 2. Prompt Instruksi (Meminta JSON, bukan CSV)
        prompt_scan = """
        Baca tabel di gambar buku catatan ini. Ekstrak datanya baris demi baris.
        Keluarkan HANYA dalam format JSON array yang berisi objek/dictionary.
        Jangan gunakan awalan/akhiran apapun, jangan ada markdown ```json.
        Pastikan angka uang ditulis sebagai integer tanpa tanda baca.
        Anda WAJIB menggunakan HANYA 4 nama key berikut untuk setiap baris:
        "tanggal", "deskripsi", "pemasukan", "pengeluaran".
        
        Contoh output yang benar:
        [
            {"tanggal": "2026-04-01", "deskripsi": "Iuran RT", "pemasukan": 50000, "pengeluaran": 0},
            {"tanggal": "2026-04-02", "deskripsi": "Beli Sapu", "pemasukan": 0, "pengeluaran": 15000}
        ]
        """
        
        # 3. Minta Gemini Membaca Gambar
        _require_gemini()
        data_ekstraksi = _call_gemini(prompt_scan, img_pil)
        
        return {
            "status": "success",
            "message": "Halaman berhasil discan. Silakan preview di Front-End.",
            "data": data_ekstraksi
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gagal scan halaman")
        raise HTTPException(status_code=500, detail=f"Gagal scan halaman: {str(exc)}")


@app.post("/scan-halaman-split")
async def scan_halaman_split(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_pil = _read_image(contents)

        prompt_scan = """
        Baca tabel di gambar buku catatan ini. Ekstrak datanya baris demi baris.
        Keluarkan HANYA dalam format JSON array yang berisi objek/dictionary.
        Jangan gunakan awalan/akhiran apapun, jangan ada markdown ```json.
        Pastikan angka uang ditulis sebagai integer tanpa tanda baca.
        Anda WAJIB menggunakan HANYA 4 nama key berikut untuk setiap baris:
        "tanggal", "deskripsi", "pemasukan", "pengeluaran".
        
        Contoh output yang benar:
        [
            {"tanggal": "2026-04-01", "deskripsi": "Iuran RT", "pemasukan": 50000, "pengeluaran": 0},
            {"tanggal": "2026-04-02", "deskripsi": "Beli Sapu", "pemasukan": 0, "pengeluaran": 15000}
        ]
        """

        _require_gemini()
        left, right = _split_left_right(img_pil)
        left_data = _call_gemini(prompt_scan, left)
        right_data = _call_gemini(prompt_scan, right)

        combined: List[Dict[str, Any]] = []
        if isinstance(left_data, list):
            combined.extend(left_data)
        if isinstance(right_data, list):
            combined.extend(right_data)

        return {
            "status": "success",
            "message": "Halaman berhasil discan (split kiri/kanan).",
            "data": combined,
            "parts": {
                "left": left_data,
                "right": right_data,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gagal scan halaman split")
        raise HTTPException(status_code=500, detail=f"Gagal scan halaman split: {str(exc)}")

# ==========================================
# SKEMA DATA UNTUK MENERIMA JSON DARI FRONT-END
# ==========================================
class ExcelRequest(BaseModel):
    data_tabel: List[Dict[str, Any]]

# ==========================================
# FITUR 1B: GENERATE EXCEL SAAT KLIK "SELESAI"
# ==========================================
@app.post("/generate-excel")
async def buat_excel_final(payload: ExcelRequest):
    try:
        # 1. Ambil data JSON gabungan yang dikirim dari Front-End
        data_mentah = payload.data_tabel
        
        # Jika kosong, tolak
        if not data_mentah:
            return {"status": "error", "message": "Data tidak boleh kosong"}

        # 2. Ubah JSON menjadi DataFrame Pandas
        df = pd.DataFrame(data_mentah)
        
        # 3. Tulis DataFrame langsung ke Memory (tanpa save ke hardisk) agar lebih cepat
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Rekap Karang Taruna')
        output.seek(0)
        
        # 4. Kembalikan sebagai file Excel yang siap didownload
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
            headers={"Content-Disposition": "attachment; filename=rekap_buku_lengkap.xlsx"}
        )
        
    except Exception as exc:
        logger.exception("Gagal membuat Excel")
        raise HTTPException(status_code=500, detail=f"Gagal membuat Excel: {str(exc)}")

# ==========================================
# FITUR 2: AUTOMATED EDA & PREPROCESSING (DATA CLEANSING)
# ==========================================
@app.post("/auto-eda")
async def auto_cleaning_data(file: UploadFile = File(...)):
    try:
        # 1. Baca File Mentah yang Diupload
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename_lower.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Format file tidak didukung. Harap unggah .csv atau .xlsx")
            
        total_baris_awal = len(df)
        
        # Simpan sampel mentah (Before)
        raw_preview = df.head(5).fillna("").to_dict(orient="records")
        
        # 2. FASE PEMBERSIHAN OTOMATIS (Automated Preprocessing)
        # A. Standarisasi nama kolom (huruf kecil & garis bawah)
        df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
        
        # B. Hapus baris yang datanya kembar 100%
        df.drop_duplicates(inplace=True)
        total_baris_setelah_dedup = len(df)
        
        # C. Atasi Nilai Kosong (Missing Values)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Jika angka kosong, isi dengan angka 0
                df[col] = df[col].fillna(0)
            else:
                # Jika teks kosong, isi dengan "Tidak Tersedia"
                df[col] = df[col].fillna("Tidak Tersedia")
                
        # 3. Ambil data bersih
        cleaned_preview = df.head(5).to_dict(orient="records")
        full_cleaned_data = df.to_dict(orient="records")
        
        # 4. Kembalikan Laporan JSON beserta datanya
        return {
            "status": "success",
            "laporan_eda": {
                "total_data_awal": total_baris_awal,
                "total_data_bersih": len(df),
                "jumlah_duplikat_dihapus": total_baris_awal - total_baris_setelah_dedup
            },
            "data_preview_raw": raw_preview,
            "data_preview_clean": cleaned_preview,
            "full_data": full_cleaned_data,
            "pesan": "Data berhasil dibersihkan."
        }
        
    except Exception as exc:
        logger.exception("Gagal memproses data")
        raise HTTPException(status_code=500, detail=f"Gagal memproses data: {str(exc)}")

# ==========================================
# FITUR 3: LAPORAN AI KEUANGAN (KONSULTAN)
# ==========================================
class AIReportRequest(BaseModel):
    saldo: float
    total_in: float
    total_out: float
    latest_transactions: List[Dict[str, Any]]

@app.post("/ai-report")
async def generate_ai_report(payload: AIReportRequest):
    try:
        _require_gemini()
        
        prompt = f"""
        Anda adalah Konsultan Keuangan Ahli untuk Karang Taruna Desa.
        Tugas Anda adalah membaca ringkasan data ini dan menyusun Laporan Audit Keuangan.
        
        Data Bulan Ini:
        - Saldo Kas Saat Ini: Rp {payload.saldo:,.0f}
        - Total Pemasukan: Rp {payload.total_in:,.0f}
        - Total Pengeluaran: Rp {payload.total_out:,.0f}
        
        Transaksi Terakhir:
        {json.dumps(payload.latest_transactions, indent=2)}
        
        Buatlah laporan komprehensif dalam format Markdown yang mencakup:
        1. **Skor Kesehatan Keuangan (0-100)** beserta predikatnya (Sangat Sehat, Waspada, dsb).
        2. **Analisis Tren**: Evaluasi perbandingan total pemasukan dan pengeluaran.
        3. **Evaluasi Pengeluaran**: Perhatikan transaksi terakhir, apakah ada pemborosan?
        4. **Rekomendasi Strategis**: Berikan minimal 2 saran praktis untuk bulan depan.
        
        Gunakan Markdown yang rapi (headings, bold, lists). Gunakan bahasa yang profesional namun memotivasi pengurus pemuda Karang Taruna.
        """
        
        response = gemini_model.generate_content(
            prompt,
            request_options={"retry": Retry(maximum=0)}
        )
        return {"status": "success", "report_markdown": response.text}
    except Exception as exc:
        logger.exception("Gagal membuat laporan AI")
        raise HTTPException(status_code=500, detail=str(exc))