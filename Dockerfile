FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Version 3
WORKDIR /app

# Install dependencies
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89"
RUN pip install --no-cache-dir llama-cpp-python runpod huggingface-hub

# Pre-download the model during build
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF', filename='orpheus-3b-0.1-ft-q4_k_m.gguf')"

COPY handler.py /app/

CMD ["python", "-u", "handler.py"]
