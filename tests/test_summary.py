from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from src.model import summarize_text


def evaluate_model(
    test_path: Path,
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Evaluate the model on the test dataset using ROUGE scores.

    Args:
        test_path: Path to the test CSV file.
        model: T5 model for generation.
        tokenizer: T5 tokenizer for encoding.
        device: Device to run the model on.

    Returns:
        Dictionary of average ROUGE scores (rouge1, rouge2, rougeL).

    Raises:
        FileNotFoundError: If the test file is not found.
        ValueError: If the dataset is empty or has invalid columns.
    """
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    test_df = pd.read_csv(test_path, encoding="latin-1")
    if test_df.empty:
        raise ValueError("Test dataset is empty")
    if "text" not in test_df.columns or "ctext" not in test_df.columns:
        raise ValueError("Test dataset must contain 'text' and 'ctext' columns")
    test_df = test_df[["text", "ctext"]]
    test_df.columns = ["summary", "text"]
    test_df.dropna(inplace=True)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for _, row in test_df.iterrows():
        text = row["text"]
        ref_summary = row["summary"]
        pred_summary = summarize_text(
            text, model, tokenizer, device=device
        )
        scores = scorer.score(ref_summary, pred_summary)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    return {
        key: sum(scores) / len(scores) for key, scores in rouge_scores.items()
    }


def main() -> None:
    """Evaluate the trained model and print example summary."""
    test_path = Path(
        "C:/Users/salma/OneDrive/Desktop/UNI/Spring 2025/DSAI 353/Project_NLP/data/processed/test_summary.csv"
    )
    model_path = Path("models/checkpoints/trained_model")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    model.to(device)
    model.eval()

    # Evaluate model
    rouge_scores = evaluate_model(test_path, model, tokenizer, device)
    print("\nEvaluation Metrics:")
    print(f"Average ROUGE Scores: {rouge_scores}")

    # Example summary
    test_df = pd.read_csv(test_path, encoding="latin-1")
    test_df = test_df[["text", "ctext"]]
    test_df.columns = ["summary", "text"]
    test_df.dropna(inplace=True)
    sample_row = test_df.iloc[0]
    text = sample_row["text"]
    summary = summarize_text(text, model, tokenizer, device=device)
    print("\nExample Summary:")
    print(f"Input Text: {text[:100]}...")
    print(f"Generated Summary: {summary}")


if __name__ == "__main__":
    main()