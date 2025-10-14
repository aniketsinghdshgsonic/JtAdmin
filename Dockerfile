



FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3-dev \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    pkg-config \
    libssl-dev \
    sqlite3 \
    file \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_CACHE_DISABLE=1

WORKDIR /app

RUN pip install --upgrade pip==23.2.1 setuptools==68.0.0 wheel==0.41.2

ENV PIP_TIMEOUT=1000
ENV PIP_RETRIES=3

RUN pip install --timeout=1000 \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

RUN pip install --no-cache-dir --timeout=1000 \
    flask==2.3.3 \
    flask-cors==4.0.0 \
    werkzeug==2.3.7 \
    mysql-connector-python==8.1.0 \
    PyMySQL==1.1.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    numpy==1.24.3 \
    pytest==7.4.2 \
    pytest-flask==1.3.0 \
    gunicorn==21.2.0 \
    orjson==3.9.7 \
    regex==2023.8.8

RUN pip install --timeout=1000 \
    "marshmallow==3.20.1" \
    "environs==10.3.0"

RUN pip install --timeout=1000 \
    "grpcio==1.57.0" \
    "grpcio-tools==1.57.0" \
    "protobuf==4.24.4"

RUN pip install --timeout=1000 \
    "ujson==5.8.0" \
    "pandas==2.0.3" \
    "pymilvus==2.3.4"

RUN pip install --timeout=1000 \
    "click==8.1.7" \
    "itsdangerous==2.1.2" \
    "jinja2==3.1.2" \
    "markupsafe==2.1.3"

RUN pip install --timeout=1000 \
    "scipy==1.10.1" \
    "scikit-learn==1.3.0"

RUN pip install --timeout=1000 \
    "huggingface-hub==0.34.4" \
    "tokenizers==0.22.0" \
    "transformers==4.56.1"

RUN pip install --timeout=1000 \
    "accelerate==1.10.1" \
    "peft==0.17.1" \
    "datasets==4.0.0" \
    "bitsandbytes==0.47.0" \
    "python-multipart==0.0.9"

RUN pip install --timeout=1000 \
    "nltk==3.8.1" \
    "sentencepiece==0.2.0" \
    "pillow>=5.3.0,!=8.3.*" \
    "tqdm>=4.32.1" \
    "sentence-transformers==2.7.0"



# Temporarily add stubs to library path for build
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install llama-cpp-python with CUDA support
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Install llama-cpp-python with CUDA support (use RUN with inline env var)
RUN LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_NATIVE=on" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python

# Remove stubs from library path (runtime will use real drivers)
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Verify installations
# RUN python -c "import llama_cpp; print('llama-cpp-python imported successfully')"
RUN python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
RUN python -c "import pymilvus; print('pymilvus imported successfully')"
RUN python -c "import transformers; print('transformers imported successfully')"
RUN python -c "import peft; print('peft imported successfully')"
RUN python -c "import bitsandbytes; print('bitsandbytes imported successfully')"

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/data /app/temp /app/.cache /app/ai-model && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser .env .

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN sed -i 's/\r$//' ./scripts/startup.sh && \
    chmod +x ./scripts/startup.sh && \
    chmod 644 ./ai-model/* 2>/dev/null || true && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["bash", "-c", "./scripts/startup.sh || (echo 'Startup failed' && sleep 300)"]