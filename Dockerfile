FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Не запускаем python handler.py — это делает RunPod
CMD ["python3"]
