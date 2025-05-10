from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import reformat_df_cleaned, tokenize_dataset


def cosine_sim_torch(pred_emb: torch.Tensor, ref_emb: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        pred_emb: Predicted embedding tensor.
        ref_emb: Reference embedding tensor.

    Returns:
        Cosine similarity score.
    """
    pred_norm = F.normalize(pred_emb, p=2, dim=-1)
    ref_norm = F.normalize(ref_emb, p=2, dim=-1)
    return torch.sum(pred_norm * ref_norm).item()


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    raw_dataset: Dataset,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> List[Dict]:
    """Evaluate the QA generation model using cosine similarity.

    Args:
        model: Trained Mistral model.
        tokenizer: Tokenizer for decoding.
        raw_dataset: Raw validation dataset.
        dataloader: DataLoader for tokenized validation data.
        device: Device to run the model on (e.g., 'cuda:0' or 'cpu').

    Returns:
        List of dictionaries containing evaluation results.

    Raises:
        RuntimeError: If evaluation fails due to device or model issues.
    """
    model.eval()
    results = []
    total = 0
    total_score = 0.0

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating QA Generation Model")):
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    temperature=0.7,
                )

            gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            labels = batch["labels"]
            labels[labels == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for j, (gen, ref) in enumerate(zip(gen_texts, ref_texts)):
                idx = i * dataloader.batch_size + j
                if idx >= len(raw_dataset):
                    continue
                example = raw_dataset[idx]

                gen_ids = tokenizer(
                    gen, return_tensors="pt", truncation=True, padding=True
                )["input_ids"].float()
                ref_ids = tokenizer(
                    ref, return_tensors="pt", truncation=True, padding=True
                )["input_ids"].float()

                max_len = max(gen_ids.shape[-1], ref_ids.shape[-1])
                gen_ids = F.pad(gen_ids, (0, max_len - gen_ids.shape[-1]))
                ref_ids = F.pad(ref_ids, (0, max_len - ref_ids.shape[-1]))

                score = cosine_sim_torch(gen_ids.squeeze(), ref_ids.squeeze())

                total += 1
                total_score += score

                results.append(
                    {
                        "context": example["context"],
                        "question": example["question"],
                        "reference_answer": example["answers"],
                        "generated_answer": gen.strip(),
                        "cosine_similarity": score,
                    }
                )

        except RuntimeError as e:
            print(f"Error during evaluation: {e}")
            torch.cuda.empty_cache()
            continue

    avg_score = total_score / total if total > 0 else 0.0
    print(f"Average Cosine Similarity over {total} examples: {avg_score:.4f}")
    return results


def main() -> None:
    """Evaluate the trained QA generation model."""
    val_path = Path("Project_NLP/data/processed/val_qa_gen.csv")
    model_path = Path("models/mistral_qa_gen")
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    model.to(device)
    model.eval()

    # Prepare validation dataset
    df = pd.read_csv(val_path)
    if df.empty:
        raise ValueError("Validation dataset is empty")
    if not all(col in df.columns for col in ["context", "question", "answers"]):
        raise ValueError("Dataset must contain 'context', 'question', 'answers' columns")

    df_cleaned = df[["context", "question", "answers"]]
    df_cleaned = df_cleaned[df_cleaned["question"].str.endswith("?", na=False)].reset_index(
        drop=True
    )
    val_df_qa_gen = reformat_df_cleaned(df_cleaned, task="qa_generation")
    val_dataset = Dataset.from_pandas(val_df_qa_gen)

    # Tokenize and create dataloader
    val_tokenized = tokenize_dataset(val_dataset, tokenizer)
    val_dataloader = torch.utils.data.DataLoader(
        val_tokenized,
        batch_size=1,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Evaluate model
    results = evaluate_model(model, tokenizer, val_dataset, val_dataloader, device)

    # Print sample results
    for r in range(min(5, len(results))):
        print(f"\nSample {r+1}:")
        print(f"Context: {results[r]['context'][:100]}...")
        print(f"Question: {results[r]['question']}")
        print(f"Reference Answer: {results[r]['reference_answer']}")
        print(f"Generated Answer: {results[r]['generated_answer']}")
        print(f"Cosine Similarity: {results[r]['cosine_similarity']:.4f}")


if __name__ == "__main__":
    main()