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

# Endpoint configurations
vllm_model_id = "data-lora"
ngrok_url = "https://419e-34-124-139-137.ngrok-free.app"         # For answering
ngrok_url_qna = "https://43cb-34-87-16-127.ngrok-free.app"    # For Q&A generation

@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
def process(
    request: Request,
    user_input: str = Form(...),
    question: str = Form(default=""),
    action: str = Form(...)
):
    transcript = process_input(user_input)
    output = ""

    if action == "summarize":
        summarize_url = "https://276d-34-87-16-127.ngrok-free.app/summarize"
        try:
            response = requests.post(summarize_url, json={"text": transcript})
            response.raise_for_status()
            summary = response.json()
            output = f"<strong>Summary:</strong> {summary['summary']}..."
        except requests.RequestException as e:
            output = f"<strong>Error during summarization:</strong> {str(e)}"

    elif action == "qna":
        try:
            response = requests.post(
                f"{ngrok_url_qna}/generate-qa",
                json={"context": transcript, "max_tokens": 512}
            )
            response.raise_for_status()
            result = response.json()
            full_text = result.get("qa", "").strip()

            qa_pairs = re.findall(r"(Q\d?:\s?.+?\nA\d?:\s?.+)", full_text, re.IGNORECASE)
            if qa_pairs:
                output = "<strong>Q&A Output:</strong><br>" + "<br>".join([pair.replace("\n", "<br>") for pair in qa_pairs])
            else:
                output = "<strong>Q&A Output:</strong><br>" + full_text.replace("\n", "<br>")
        except requests.RequestException as e:
            output = f"<strong>Error generating Q&A:</strong> {str(e)}"

    elif action == "answer":
        if not question:
            output = "Please enter a question."
        else:
            prompt = (
                f"Context: {transcript}\n"
                f"Question: {question}\n"
                f"Instruction: Answer concisely in 1-5 words based on the context.\n"
                f"Answer: "
            )
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
                answers = re.findall(r"Answer\s*:\s*(.*)", full_text, re.IGNORECASE)
                answer = answers[-1].strip() if answers else full_text
                output = f"<strong>Answer:</strong> {answer}"
            except requests.RequestException as e:
                output = f"<strong>Error generating answer:</strong> {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "output":output})