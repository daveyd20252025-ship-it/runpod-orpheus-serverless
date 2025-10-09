FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install llama-cpp-python with CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89"
RUN pip install --no-cache-dir llama-cpp-python runpod huggingface-hub

COPY handler.py /app/

CMD ["python", "-u", "handler.py"]
