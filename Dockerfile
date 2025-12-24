# Dockerfile for Stock Agent (Rebuild Trigger 1)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install -e .

EXPOSE 7860
CMD ["uvicorn", "src.api.index:app", "--host", "0.0.0.0", "--port", "7860"]