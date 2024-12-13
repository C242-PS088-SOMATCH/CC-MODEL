# Gunakan image Python yang resmi
FROM python:3.9-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Copy file requirements.txt dan install dependensi
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy sisa aplikasi (termasuk app.py dan model) ke dalam container
COPY . /app

# Expose port Flask (default port 8080)
EXPOSE 8080

# Set environment variable agar Flask berjalan di host 0.0.0.0 dan port 8080
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Jalankan Flask app
CMD ["python", "app.py"]
