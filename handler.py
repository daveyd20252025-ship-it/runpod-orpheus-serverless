import runpod
from llama_cpp import Llama
import os

MODEL_PATH = "/root/.cache/huggingface/hub/models--isaiahbjork--orpheus-3b-0.1-ft-Q4_K_M-GGUF/snapshots/af161b11022b996f8ae2f54d79b8ff71c5a3fb58/orpheus-3b-0.1-ft-q4_k_m.gguf"

llm = None

def handler(event):
    global llm
    
    # Load model on first request
    if llm is None:
        print("Loading model...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=-1,
            n_threads=4,
            verbose=False,
            use_mmap=False  # THIS IS KEY - don't use mmap
        )
        print("Model loaded!")
    
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "Hello")
        max_tokens = input_data.get("max_tokens", 100)
        
        response = llm(prompt, max_tokens=max_tokens, temperature=0.7)
        
        return {"output": response["choices"][0]["text"]}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
