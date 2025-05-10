from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import re
from logic import process_input

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# vLLM endpoint configuration
vllm_model_id = "data-lora"
ngrok_url = "https://b458-34-90-55-87.ngrok-free.app"

@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
def process(request: Request, user_input: str = Form(...), question: str = Form(default=""), action: str = Form(...)):
    transcript = process_input(user_input)
    output = ""

    if action == "summarize":
        url = "https://abb1-35-197-132-141.ngrok-free.app/summarize"
        data = {"text": transcript}
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            summary = response.json()
            output = f"Summary: {summary['summary']}..."
        except requests.RequestException as e:
            output = f"Error during summarization: {str(e)}"

    elif action == "qna":
        output = f"Generated Q&A from: {transcript[:80]}..."

    elif action == "answer":
        context = transcript

        if not question:
            output = "Please enter a question."
        else:
            prompt = (
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Instruction: Answer concisely in 1-5 words based on the context.\n"
                f"Answer: "
            )

            import re

        try:
            response = requests.post(
                f"{ngrok_url}/v1/completions",
                json={
                    "model": vllm_model_id,
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            result = response.json()
            full_text = result.get("choices", [{}])[0].get("text", "").strip()
            print("Model raw output:", full_text)  # Debug print

            # Find all 'Answer: ...' patterns
            answers = re.findall(r"Answer\s*:\s*(.*)", full_text, re.IGNORECASE)
            if answers:
                answer = answers[-1].strip()  # Take the last one
                output = f"Answer: {answer}"
               
            else:
                output = f"Answer: {full_text}"
        except requests.RequestException as e:
            output = f"Error generating answer: {str(e)}"

    print("rrr",output)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "output": output}
    )