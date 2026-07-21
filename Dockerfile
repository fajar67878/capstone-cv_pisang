FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file proyek
COPY . .

# Koyeb menggunakan port 8000 secara default
EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]