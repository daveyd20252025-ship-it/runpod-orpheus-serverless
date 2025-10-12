import runpod
from llama_cpp import Llama
import os

MODEL_PATH = "/root/.cache/huggingface/hub/models--isaiahbjork--orpheus-3b-0.1-ft-Q4_K_M-GGUF/snapshots/af161b11022b996f8ae2f54d79b8ff71c5a3fb58/orpheus-3b-0.1-ft-q4_k_m.gguf"

llm = None

def handler(event):
    global llm
    
    if llm is None:
        print("Loading model...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,              # ✅ --ctx-size 4096
            n_gpu_layers=-1,         # ✅ --gpu-layers 999 (all)
            n_batch=512,             # ✅ --batch-size 512
            n_threads=8,             # ✅ --threads 8
            use_mmap=False,          # ✅ --no-mmap
            verbose=False
            # ❌ ubatch-size - not available in Python lib
            # ❌ threads-batch - not available
            # ❌ flash-attn - not available
        )
        print("Model loaded!")
    
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "")
        max_tokens = input_data.get("max_tokens", 256)
        
        # Generation parameters (per-request)
        response = llm(
            prompt, 
            max_tokens=max_tokens,
            temperature=0.7,        # ✅ --temp 0.7
            top_p=0.9,              # ✅ --top-p 0.9
            top_k=40,               # ✅ --top-k 40
            repeat_penalty=1.05     # ✅ --repeat-penalty 1.05
        )
        
        return {"output": response["choices"][0]["text"]}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
