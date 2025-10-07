FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp WITH CUDA for RTX 3090/4090
WORKDIR /app/llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86;89 \
    -DLLAMA_CUDA=ON \
    && cmake --build build --config Release -j$(nproc)

# Install Python dependencies
WORKDIR /app
RUN pip install --no-cache-dir runpod huggingface-hub requests

# Copy handler
COPY runpod_handler.py /app/

CMD ["python", "-u", "runpod_handler.py"]
