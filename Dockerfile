FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git build-essential cmake python3 python3-pip python3-venv curl wget pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone llama.cpp (CPU-only build, shallow clone for faster download)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp CPU backend (no CUDA/HIP)
WORKDIR /app/llama.cpp
RUN cmake -B build -DGGML_CUDA=off -DGGML_HIPBLAS=off && cmake --build build -j$(nproc)

# Install Python dependencies
RUN pip install --upgrade pip && pip install runpod requests huggingface-hub

# Copy handler into container
COPY runpod_handler.py /app/

# Final setup
WORKDIR /app
EXPOSE 8000

CMD ["python3", "runpod_handler.py"]

