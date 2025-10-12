import runpod
import subprocess
import requests
import time
import os

MODEL_PATH = "/root/.cache/huggingface/hub/models--isaiahbjork--orpheus-3b-0.1-ft-Q4_K_M-GGUF/snapshots/af161b11022b996f8ae2f54d79b8ff71c5a3fb58/orpheus-3b-0.1-ft-q4_k_m.gguf"
SERVER_PORT = 5006

# Start llama.cpp server with RTX 4090 optimizations on import
print("Starting optimized llama.cpp server...")
server_process = subprocess.Popen([
    "/app/llama.cpp/build/bin/server",
    "-m", MODEL_PATH,
    "--host", "0.0.0.0",
    "--port", str(SERVER_PORT),
    "--ctx-size", "4096",
    "--gpu-layers", "999",
    "--batch-size", "512",
    "--ubatch-size", "128",
    "--threads", "8",
    "--threads-batch", "8",
    "--no-mmap",
    "--flash-attn",
    "--repeat-penalty", "1.05",
    "--top-k", "40",
    "--top-p", "0.9",
    "--temp", "0.7",
    "--log-disable"
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for server to be ready
time.sleep(10)
print("Server ready with flash attention!")

def handler(event):
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        max_tokens = input_data.get("max_tokens", 256)
        
        # Call the llama.cpp server
        response = requests.post(
            f"http://localhost:{SERVER_PORT}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens
            },
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"Server error: {response.text}"}
        
        result = response.json()
        return {"output": result.get("content", "")}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
