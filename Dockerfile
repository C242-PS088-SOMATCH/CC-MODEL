# Gunakan image Python
FROM python:3.9-slim

# Tetapkan direktori kerja
WORKDIR /app

# Salin file proyek
COPY . .

# Instal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Tetapkan port yang akan digunakan
ENV PORT=8080

# Gunicorn akan digunakan untuk menjalankan aplikasi
CMD exec gunicorn -w 4 -b :$PORT --timeout 3000 app:app
