FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git build-essential cmake python3 python3-pip curl wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp
RUN cmake -B build -DGGML_CUDA=on -DGGML_HIPBLAS=off && cmake --build build -j$(nproc)
RUN pip install runpod requests huggingface-hub
COPY runpod_handler.py /app/
WORKDIR /app
EXPOSE 8000
CMD ["python3", "runpod_handler.py"]
