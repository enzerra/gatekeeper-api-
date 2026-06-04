FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh source code
COPY . .

# HuggingFace Spaces menggunakan port 7860
ENV PORT=7860

# Jalankan Uvicorn FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
