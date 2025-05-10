from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import torch
import torch.nn as nn
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Any

def cosine_sim_torch(pred_emb: torch.Tensor, ref_emb: torch.Tensor) -> float:
    """Calculate cosine similarity between two embeddings using PyTorch.

    This function computes the cosine similarity between two input tensors by normalizing
    them along their last dimension and calculating their dot product. The result is a
    scalar value representing the cosine similarity.

    Args:
        pred_emb (torch.Tensor): Predicted embedding tensor.
        ref_emb (torch.Tensor): Reference embedding tensor.

    Returns:
        float: Cosine similarity score between the two embeddings.


    """
    pred_norm = nn.functional.normalize(pred_emb, p=2, dim=-1)
    ref_norm = nn.functional.normalize(ref_emb, p=2, dim=-1)
    return torch.sum(pred_norm * ref_norm).item()


def evaluate_model_with_cosine_sim(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_df: pd.DataFrame,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 5
) -> List[Dict[str, Any]]:
    """Evaluate a model using cosine similarity between generated and reference answers.

    This function evaluates a transformer model by generating answers to questions based on
    provided contexts, then compares the generated answers to reference answers using cosine
    similarity of their sentence embeddings. It processes a specified number of batches from
    the dataloader and returns detailed results for each sample.

    Args:
        model (PreTrainedModel): The transformer model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        dataset_df (pd.DataFrame): DataFrame containing context, question, and reference answers.
        dataloader (DataLoader): DataLoader for iterating over the dataset.
        device (torch.device): Device (e.g., 'cuda:0') to run the model on.
        num_batches (int, optional): Number of batches to process. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each sample,
            with keys: 'context', 'question', 'reference_answer', 'generated_answer', and
            'cosine_similarity'.

    Raises:
        Exception: If embedding computation fails, the error is caught, a score of 0.0 is assigned,
            and evaluation continues.


    """
    model.eval()
    results = []
    total = 0
    total_score = 0.0
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating with Cosine Similarity")):
        if i >= num_batches:
            break

        start_idx = i * dataloader.batch_size
        end_idx = min((i + 1) * dataloader.batch_size, len(dataset_df))
        batch_df = dataset_df.iloc[start_idx:end_idx]

        for j, (_, row) in enumerate(batch_df.iterrows()):
            context = row["context"]
            question = row["question"]
            reference_answer = row["answers"]
            prompt = (
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Instruction: Answer concisely in 1-5 words based on the context.\n"
                f"Answer: "
            )

            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
            generated_text = " ".join(generated_text.split()[:5])

            try:
                gen_emb = torch.tensor(sentence_model.encode(generated_text, convert_to_numpy=True))
                ref_emb = torch.tensor(sentence_model.encode(reference_answer, convert_to_numpy=True))
                score = cosine_sim_torch(gen_emb, ref_emb)
            except Exception as e:
                print(f"Error computing embeddings for index {start_idx + j}: {e}")
                score = 0.0

            results.append({
                "context": context,
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_text,
                "cosine_similarity": score
            })

            total += 1
            total_score += score

    avg_score = total_score / total if total > 0 else 0.0
    print(f"\nAverage Cosine Similarity over {total} samples: {avg_score:.4f}")
    return results