# # Install required packages
# # !pip install fastapi uvicorn nest-asyncio pyngrok transformers torch --quiet

# # Fix protobuf compatibility
# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# import torch
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from fastapi import FastAPI
# from pydantic import BaseModel
# import nest_asyncio
# from pyngrok import ngrok
# import uvicorn

# # (Optional but Recommended) Set your HF token (if model is private)
# # os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxx"
# # hf_token = os.environ.get("HF_TOKEN")
# hf_token = None  # If your model is public, keep this as None

# # Define repository and version
# repo_id = "SalmaSherif202200622/Summarization_NLP_model"

# # Load model and tokenizer
# model = T5ForConditionalGeneration.from_pretrained(repo_id, use_auth_token=hf_token)
# tokenizer = T5Tokenizer.from_pretrained(repo_id, use_auth_token=hf_token)

# # Set device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model.to(device)
# model.eval()

# # FastAPI app
# app = FastAPI()

# # Request schema
# class TextRequest(BaseModel):
#     text: str

# # Summarize function
# def summarize_text(text, max_length=128, num_beams=4):
#     inputs = tokenizer(
#         text,
#         max_length=512,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_length=max_length,
#             num_beams=num_beams,
#             early_stopping=True
#         )
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return summary

# # POST endpoint
# @app.post("/summarize")
# async def summarize(request: TextRequest):
#     text = request.text
#     summary = summarize_text(text)
#     return {"summary": summary}

# # Enable running uvicorn inside notebook
# nest_asyncio.apply()

# # Open public URL using ngrok
# public_url = ngrok.connect(8000)
# print(f"Public URL: {public_url}")

# # Run FastAPI app
# uvicorn.run(app, host="0.0.0.0", port=8000)
