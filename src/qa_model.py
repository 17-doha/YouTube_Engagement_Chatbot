from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from transformers  import BitsAndBytesConfig
import pandas as pd
from peft import LoraConfig, get_peft_model

from torch.utils.data import DataLoader
from huggingface_hub import login
import torch.nn as nn
from src.train_qa import train_model

batch_size = 4


def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

train_df_qa_answer = pd.read_csv("data//processed//train_qa_answer.csv")
val_df_qa_answer = pd.read_csv("data//processed//val_qa_answer.csv")

train_dataset_qa_answer = Dataset.from_pandas(train_df_qa_answer)
val_dataset_qa_answer = Dataset.from_pandas(val_df_qa_answer)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token

train_qa_answer_tokenized = train_dataset_qa_answer.map(
    tokenize_function,
    batched=True,
    remove_columns=["prompt", "context", "question", "answers"]
)
val_qa_answer_tokenized = val_dataset_qa_answer.map(
    tokenize_function,
    batched=True,
    remove_columns=["prompt", "context", "question", "answers"]
)

train_qa_answer_tokenized.set_format("torch")
val_qa_answer_tokenized.set_format("torch")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.float16
)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


