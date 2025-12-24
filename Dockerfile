FROM intel/intel-extension-for-pytorch:2.1.10-xpu
WORKDIR /app

USER root

RUN rm -f /etc/apt/sources.list.d/*intel* && \
    apt-get update && apt-get install -y \
    git \
    wget \
    gnupg \
    clinfo \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
COPY . .
RUN pip install -e .
RUN pip install --no-cache-dir torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
EXPOSE 7860
CMD ["uvicorn", "src.api.index:app", "--host", "0.0.0.0", "--port", "7860"]