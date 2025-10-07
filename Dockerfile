FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set CUDA compile flags
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89"
ENV FORCE_CMAKE=1

# Install llama-cpp-python with CUDA support
RUN pip install --no-cache-dir llama-cpp-python --force-reinstall --upgrade

# Install required packages
RUN pip install --no-cache-dir runpod huggingface_hub

# Copy application files
COPY . .

# Add a test to verify imports work
RUN python -c "import huggingface_hub; import runpod; print('Dependencies OK')"

CMD ["python", "-u", "runpod_handler.py"]
