from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import re

app = FastAPI()

vllm_model_id = "data-lora"
ngrok_url = "https://137b-34-16-198-36.ngrok-free.app"  

# Request schema
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        response = requests.post(f"{ngrok_url}/v1/completions", json={
            "model": vllm_model_id,
            "prompt": request.prompt,
            "max_tokens": 200,
            "temperature": 0.3
        })

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        result = response.json()
        full_text = result.get("text", "").strip()
        match = re.search(r"(?:\n|^)Answer(?: based on context)?:\s*(.*)", full_text)
        if match:
            result = match.group(1).strip()
        else:
            result = ""

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
