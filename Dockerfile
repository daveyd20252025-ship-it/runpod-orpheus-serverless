FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    git build-essential cmake python3 python3-pip python3-venv curl wget pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone llama.cpp
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp CPU backend
WORKDIR /app/llama.cpp
RUN cmake -B build -DGGML_CUDA=off -DGGML_HIPBLAS=off && cmake --build build -j$(nproc)

# Install Python deps
WORKDIR /app
RUN pip install --upgrade pip && \
    pip install runpod requests huggingface-hub

# Copy handler
COPY runpod_handler.py /app/

EXPOSE 8000
CMD ["python3", "runpod_handler.py"]
