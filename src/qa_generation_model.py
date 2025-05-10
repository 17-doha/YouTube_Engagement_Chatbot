from typing import Any, Dict, Union

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

def reformat_df_cleaned(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Reformat DataFrame for QA generation or answering tasks.

    Args:
        df: Input DataFrame with 'context', 'question', and 'answers' columns.
        task: Task type, either 'qa_generation' or 'qa_answering'.

    Returns:
        Reformatted DataFrame with 'prompt' column.

    Raises:
        ValueError: If task is not 'qa_generation' or 'qa_answering'.
    """
    df_reformatted = df.copy()
    df_reformatted["parsed_answer"] = df_reformatted["answers"].apply(
        lambda x: x.strip("[]").split("'")[1] if x else "not enough information"
    )

    if task == "qa_generation":
        df_reformatted["prompt"] = (
            "### Context:\n"
            + df_reformatted["context"]
            + "\n\n"
            + "### Instruction:\nGenerate a question and its answer based on the context.\n\n"
            + "### Output:\n**Question**: "
            + df_reformatted["question"]
            + "\n"
            + "**Answer**: "
            + df_reformatted["parsed_answer"]
        )
    elif task == "qa_answering":
        df_reformatted["prompt"] = (
            "### Context:\n"
            + df_reformatted["context"]
            + "\n\n"
            + "### Question:\n"
            + df_reformatted["question"]
            + "\n\n"
            + "### Instruction:\nProvide the answer to the question based on the context.\n\n"
            + "### Answer:\n"
            + df_reformatted["parsed_answer"]
        )
    else:
        raise ValueError("Task must be 'qa_generation' or 'qa_answering'")

    return df_reformatted.drop(columns=["parsed_answer"])


def prepare_datasets(data_path: str, test_size: float = 0.2) -> tuple[Dataset, Dataset]:
    """Load and prepare datasets for QA generation.

    Args:
        data_path: Path to the input CSV file.
        test_size: Proportion of data for validation split.

    Returns:
        Tuple of train and validation Datasets.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the dataset is empty or lacks required columns.
    """
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty")
    if not all(col in df.columns for col in ["context", "question", "answers"]):
        raise ValueError("Dataset must contain 'context', 'question', 'answers' columns")

    df_cleaned = df[["context", "question", "answers"]]
    df_cleaned = df_cleaned[df_cleaned["question"].str.endswith("?", na=False)].reset_index(
        drop=True
    )

    train_df, val_df = train_test_split(df_cleaned, test_size=test_size, random_state=42)
    train_df_qa_gen = reformat_df_cleaned(train_df, task="qa_generation")
    val_df_qa_gen = reformat_df_cleaned(val_df, task="qa_generation")

    return Dataset.from_pandas(train_df_qa_gen), Dataset.from_pandas(val_df_qa_gen)


def tokenize_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 512
) -> Dataset:
    """Tokenize the dataset for training.

    Args:
        dataset: Input Dataset with 'prompt' column.
        tokenizer: Tokenizer for encoding text.
        max_length: Maximum token length.

    Returns:
        Tokenized Dataset with 'input_ids', 'attention_mask', and 'labels'.
    """

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "context", "question", "answers"],
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def setup_model_and_tokenizer(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    use_quantization: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize the Mistral model and tokenizer with quantization and LoRA.

    Args:
        model_name: Name of the pretrained Mistral model.
        use_quantization: Whether to use 4-bit quantization.

    Returns:
        Tuple of initialized model and tokenizer.

    Raises:
        RuntimeError: If model or tokenizer loading fails.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            if use_quantization
            else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model or tokenizer: {e}")


def create_dataloader(
    dataset: Dataset, tokenizer: AutoTokenizer, batch_size: int = 1
) -> DataLoader:
    """Create a DataLoader for the tokenized dataset.

    Args:
        dataset: Tokenized Dataset.
        tokenizer: Tokenizer for data collation.
        batch_size: Batch size for DataLoader.

    Returns:
        DataLoader for training or evaluation.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )