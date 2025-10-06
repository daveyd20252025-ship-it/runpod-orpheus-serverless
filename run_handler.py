from runpod.serverless import start
import subprocess, requests, time, os
from huggingface_hub import hf_hub_download

MODEL_REPO = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
MODEL_FILE = "orpheus-3b-0.1-ft-q4_k_m.gguf"
MODEL_PATH = f"/models/{MODEL_FILE}"
PORT = 8000

os.makedirs("/models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="/models")

print("Starting llama.cpp server...")
subprocess.Popen([
    "/app/llama.cpp/build/bin/server",
    "-m", MODEL_PATH,
    "--ctx-size", "4096",
    "--threads", "8",
    "--host", "0.0.0.0",
    "--port", str(PORT)
])

time.sleep(5)

def handler(event):
    text = event["input"].get("text", "Hello from Orpheus!")
    payload = {
        "model": "orpheus-3b-0.1-ft-q4_k_m",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 256
    }
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload)
    return r.json()

start({"handler": handler})

