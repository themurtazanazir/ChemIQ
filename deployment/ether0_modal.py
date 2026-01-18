"""
Modal deployment for ether0 (FutureHouse 24B chemistry model) - Unquantized
Exposes OpenAI SDK-compatible API at /v1/chat/completions.

Run with: modal deploy deployment/ether0_modal.py

Usage with OpenAI SDK:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://themurtazanazir--ether0-inference-web.modal.run/v1",
        api_key="not-needed"
    )
    response = client.chat.completions.create(
        model="ether0",
        messages=[{"role": "user", "content": "SMILES for water?"}]
    )

Usage with curl:
    curl -X POST https://themurtazanazir--ether0-inference-web.modal.run/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "ether0", "messages": [{"role": "user", "content": "SMILES for water?"}]}'
"""

import modal
import time
from uuid import uuid4

# --- Configuration ---
MODEL_ID = "futurehouse/ether0"
MODEL_REVISION = "main"
GPU_CONFIG = "A100-80GB"  # 24B FP16 needs ~48GB VRAM
VOLUME_NAME = "ether0-model-cache"
CACHE_PATH = "/model-cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "huggingface_hub>=0.21.0",
        "vllm>=0.4.0",
        "sentencepiece",
        "protobuf",
        "fastapi",
    )
    .env({"HF_HOME": CACHE_PATH})
)

app = modal.App("ether0-inference", image=image)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def download_model_to_volume():
    """Download model weights to volume (called on first run)."""
    import os
    from huggingface_hub import snapshot_download
    
    model_path = os.path.join(CACHE_PATH, "hub", f"models--{MODEL_ID.replace('/', '--')}")
    if os.path.exists(model_path):
        print(f"Model already cached at {model_path}")
        return
    
    print(f"Downloading {MODEL_ID} to volume...")
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        ignore_patterns=["*.gguf", "*.ggml"],
    )
    print("Download complete!")


# --- OpenAI SDK-compatible Web Endpoint ---
@app.function(
    gpu=GPU_CONFIG,
    timeout=600,
    scaledown_window=300,
    volumes={CACHE_PATH: model_volume},
)
@modal.asgi_app()
def web():
    """FastAPI app with OpenAI SDK-compatible /v1/chat/completions endpoint."""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List, Optional
    from vllm import LLM, SamplingParams
    
    # Load model at startup
    download_model_to_volume()
    model_volume.commit()
    
    print(f"Loading {MODEL_ID} for web endpoint...")
    llm = LLM(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    print("Model loaded!")
    
    fastapi_app = FastAPI(title="Ether0 OpenAI-Compatible API")
    
    class Message(BaseModel):
        role: str
        content: str
    
    class ChatRequest(BaseModel):
        model: str = "ether0"
        messages: List[Message]
        max_tokens: Optional[int] = 10000
        temperature: Optional[float] = 0.0
        top_p: Optional[float] = 0.95
    
    @fastapi_app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        """OpenAI-compatible chat completions endpoint."""
        # Extract last user message as prompt
        prompt = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                prompt = msg.content
                break
        
        if not prompt:
            return {"error": "No user message found"}
        
        # Generate
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        outputs = llm.generate([prompt], params)
        output = outputs[0]
        
        # Build OpenAI-format response
        completion_text = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        
        return {
            "id": f"chatcmpl-{uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "ether0",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text,
                },
                "finish_reason": output.outputs[0].finish_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    
    @fastapi_app.get("/v1/models")
    def list_models():
        """List available models (OpenAI SDK compatibility)."""
        return {
            "object": "list",
            "data": [{
                "id": "ether0",
                "object": "model",
                "created": 1700000000,
                "owned_by": "futurehouse",
            }]
        }
    
    @fastapi_app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok", "model": MODEL_ID}
    
    return fastapi_app


@app.local_entrypoint()
def main():
    """Test the deployment via OpenAI SDK."""
    print("To test the deployed endpoint, run:")
    print("""
from openai import OpenAI

client = OpenAI(
    base_url="https://themurtazanazir--ether0-inference-web.modal.run/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="ether0",
    messages=[{"role": "user", "content": "SMILES for water?"}]
)
print(response.choices[0].message.content)
""")
