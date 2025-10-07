FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git build-essential cmake python3 python3-pip python3-venv curl wget pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone llama.cpp (shallow clone for faster build)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp (CPU build only)
WORKDIR /app/llama.cpp
RUN cmake -B build -DGGML_CUDA=on -DGGML_HIPBLAS=off && cmake --build build -j$(nproc)

# Install Python libraries
RUN pip install --upgrade pip && pip install runpod requests huggingface-hub

# Copy handler
COPY runpod_handler.py /app/

WORKDIR /app
EXPOSE 8000

# Start handler
CMD ["python3", "runpod_handler.py"]
