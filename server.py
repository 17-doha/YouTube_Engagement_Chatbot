from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from logic import process_input

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
def process(request: Request, user_input: str = Form(...), action: str = Form(...)):
    transcript = process_input(user_input)
    
    if action == "summarize":
        output = f"Summary: {transcript[:100]}..."
    elif action == "qna":
        output = f"Generated Q&A from: {transcript[:80]}..."
    elif action == "answer":
        output = f"Answer based on: {transcript[:80]}..."
    else:
        output = "Invalid action."

    return output
