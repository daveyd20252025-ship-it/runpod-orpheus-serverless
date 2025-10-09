import runpod
import subprocess
import requests
import time
import os
from huggingface_hub import hf_hub_download

MODEL_REPO = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
MODEL_FILE = "orpheus-3b-0.1-ft-q4_k_m.gguf"
MODEL_PATH = f"/models/{MODEL_FILE}"
PORT = 8000

# Global variable to track if server is initialized
server_process = None
server_ready = False

def initialize_server():
    """Download model and start llama.cpp server (called once on first request)"""
    global server_process, server_ready
    
    if server_ready:
        return
    
    print("Initializing server...")
    os.makedirs("/models", exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from Hugging Face: {MODEL_FILE}")
        hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="/models")
        print("Model downloaded successfully")
    
    print("Starting llama.cpp server...")
    server_process = subprocess.Popen([
        "/app/llama.cpp/build/bin/server",
        "-m", MODEL_PATH,
        "--ctx-size", "4096",
        "--threads", "8",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--n-gpu-layers", "99"  # Use GPU for RTX 3090/4090
    ])
    
    # Wait for server to be ready
    print("Waiting for llama.cpp server to start...")
    for i in range(30):  # 30 second timeout
        try:
            response = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            if response.status_code == 200:
                print("Server is ready!")
                server_ready = True
                return
        except:
            time.sleep(1)
    
    raise Exception("llama.cpp server failed to start in 30 seconds")

def handler(event):
    """RunPod handler function"""
    try:
        # Initialize on first request (lazy loading)
        initialize_server()
        
        # Get input text
        input_data = event.get("input", {})
        text = input_data.get("text", "Hello from Orpheus!")
        
        # Make request to llama.cpp server
        payload = {
            "model": "orpheus-3b-0.1-ft-q4_k_m",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 256
        }
        
        response = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        return {"output": response.json()}
        
    except Exception as e:
        return {"error": str(e)}

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
