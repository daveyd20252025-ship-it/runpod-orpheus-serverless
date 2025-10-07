FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install llama-cpp-python with CUDA support for both 3090 (86) and 4090 (89)
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89"
ENV FORCE_CMAKE=1

COPY requirements.txt .
RUN pip install --no-cache-dir llama-cpp-python --force-reinstall --upgrade --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "handler.py"]
