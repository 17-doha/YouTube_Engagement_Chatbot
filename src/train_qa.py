import torch
from qa_model import model, tokenizer, train_qa_answer_tokenized, val_qa_answer_tokenized
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedModel
from typing import Optional

num_train_epochs = 1
batch_size = 4
device = torch.device("cuda:0")
gradient_accumulation_steps = 16
num_train_epochs = 5
learning_rate = 2e-4
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=learning_rate
)

device = torch.device("cuda:0")
output_dir = "./mistral_finetuned"

train_dataloader_qa_answer = DataLoader(
    train_qa_answer_tokenized,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
val_dataloader_qa_answer = DataLoader(
    val_qa_answer_tokenized,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)



def train_model(
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    output_dir: str,
    task_name: str,
    repo_name: str,
    access_token: Optional[str] = None
) -> None:
    """Train a transformer model and save it locally and optionally to Hugging Face Hub.

    This function trains a transformer-based model using the provided training and validation
    dataloaders. It implements gradient accumulation, periodic validation, and saves the model
    and tokenizer both locally and to the Hugging Face Hub if an access token is provided.

    Args:
        model (PreTrainedModel): The transformer model to be trained.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        output_dir (str): Directory path to save the model and tokenizer locally.
        task_name (str): Name of the task for organizing local save paths.
        repo_name (str): Name of the repository on Hugging Face Hub (e.g., 'username/repo').
        access_token (Optional[str]): Hugging Face access token for pushing to the Hub.
            Defaults to None.

    Returns:
        None

    Raises:
        RuntimeError: If a CUDA-related error occurs during training, it will be caught,
            the cache will be cleared, and training will continue.
        Exception: If pushing to Hugging Face Hub fails, an error message is printed,
            and training continues without pushing.


    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4
    )
    model.train()
    total_steps = len(train_dataloader) * 5  # num_train_epochs
    step = 0
    device = torch.device("cuda:0")
    gradient_accumulation_steps = 16

    for epoch in range(5):  # num_train_epochs
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                print(f"Loss: {loss.item()}, Requires Grad: {loss.requires_grad}")
                if loss is None or not loss.requires_grad:
                    print("Error: Loss does not require grad or is None")
                    print(f"Labels sample: {labels[0]}")
                    print(f"Outputs logits: {outputs.logits.shape}")
                loss = loss / gradient_accumulation_steps
                total_loss += loss.item()

                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                    if step % 100 == 0:
                        print(f"Epoch {epoch+1}, Step {step}/{total_steps}, Loss: {total_loss / (batch_idx + 1):.4f}")

                if (batch_idx + 1) == len(train_dataloader):
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            input_ids = val_batch["input_ids"].to(device)
                            attention_mask = val_batch["attention_mask"].to(device)
                            labels = val_batch["labels"].to(device)
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            val_loss += outputs.loss.item()
                    val_loss /= len(val_dataloader)
                    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
                    model.train()

            except RuntimeError as e:
                print(f"Error during training: {e}")
                torch.cuda.empty_cache()
                continue

        local_save_path = f"{output_dir}/{task_name}"
        model.save_pretrained(local_save_path)
        tokenizer.save_pretrained(local_save_path)
        print(f"Model and tokenizer saved locally to {local_save_path}")

        if access_token:
            try:
                model.eval()
                model.push_to_hub(repo_name, token=access_token)
                tokenizer.push_to_hub(repo_name, token=access_token)
                print(f"Model and tokenizer successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
            except Exception as e:
                print(f"Error pushing to Hugging Face Hub: {e}")
                print("Continuing without pushing to Hub...")



repo_name_qa_answer = "DohaHemdann/mistral_qa_answer3epochs"
print("Training QA Answering Model...")
train_model(model, train_dataloader_qa_answer, val_dataloader_qa_answer, output_dir, "mistral_qa_answer3epochs", repo_name_qa_answer, access_token)