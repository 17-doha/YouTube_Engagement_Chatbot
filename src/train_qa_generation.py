import gc
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import login

from .model import create_dataloader, prepare_datasets, setup_model_and_tokenizer, tokenize_dataset


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    task_name: str,
    repo_name: str,
    access_token: Optional[str],
    num_epochs: int = 5,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 16,
) -> None:
    """Train the Mistral model for QA generation.

    Args:
        model: Mistral model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        output_dir: Directory to save model checkpoints.
        task_name: Name of the task (e.g., 'mistral_qa_gen').
        repo_name: Hugging Face repository name for model upload.
        access_token: Hugging Face access token for pushing to Hub.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        gradient_accumulation_steps: Number of steps for gradient accumulation.

    Raises:
        RuntimeError: If training fails due to device or model issues.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    step = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                total_loss += loss.item()

                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                    if step % 20 == 0:
                        print(
                            f"Epoch {epoch+1}, Step {step}/{total_steps}, "
                            f"Loss: {total_loss / (batch_idx + 1):.4f}"
                        )

                if (batch_idx + 1) == len(train_dataloader):
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            input_ids = val_batch["input_ids"].to(model.device)
                            attention_mask = val_batch["attention_mask"].to(model.device)
                            labels = val_batch["labels"].to(model.device)
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                            val_loss += outputs.loss.item()
                    val_loss /= len(val_dataloader)
                    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
                    model.train()

            except RuntimeError as e:
                print(f"Error during training: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        local_save_path = output_dir / task_name
        model.save_pretrained(local_save_path)
        train_dataloader.dataset.tokenizer.save_pretrained(local_save_path)
        print(f"Model and tokenizer saved locally to {local_save_path}")

        if access_token:
            try:
                model.push_to_hub(repo_name, token=access_token)
                train_dataloader.dataset.tokenizer.push_to_hub(
                    repo_name, token=access_token
                )
                print(
                    f"Model and tokenizer pushed to Hugging Face Hub: "
                    f"https://huggingface.co/{repo_name}"
                )
            except Exception as e:
                print(f"Error pushing to Hugging Face Hub: {e}")


def main() -> None:
    """Main function to train the QA generation model."""
    train_path = Path("Project_NLP/data/processed/train_qa_gen.csv")
    val_path = Path("Project_NLP/data/processed/val_qa_gen.csv")
    output_dir = Path("models")
    task_name = "mistral_qa_gen"
    repo_name = "selsayed2003/mistral_qa_gen"
    access_token = None  # Set HF_TOKEN environment variable or provide token

    # Login to Hugging Face if token is provided
    if access_token:
        login(access_token)

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(train_path)

    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Tokenize datasets
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer)

    # Create dataloaders
    train_dataloader = create_dataloader(train_tokenized, tokenizer)
    val_dataloader = create_dataloader(val_tokenized, tokenizer)

    # Train model
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        output_dir,
        task_name,
        repo_name,
        access_token,
    )


if __name__ == "__main__":
    main()