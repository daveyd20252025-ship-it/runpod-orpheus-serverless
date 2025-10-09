import runpod
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

MODEL_REPO = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
MODEL_FILE = "orpheus-3b-0.1-ft-q4_k_m.gguf"

llm = None

def init():
    global llm
    if llm:
        return
    
    print("Downloading model from HuggingFace...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        token=os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    
    print(f"Loading model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_threads=8,
        verbose=True
    )
    print("Model loaded successfully!")

def handler(event):
    try:
        init()
        
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "Hello")
        max_tokens = input_data.get("max_tokens", 256)
        temperature = input_data.get("temperature", 0.7)
        
        print(f"Generating response for prompt: {prompt[:50]}...")
        
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "\n\n"]
        )
        
        return {"output": response["choices"][0]["text"]}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
