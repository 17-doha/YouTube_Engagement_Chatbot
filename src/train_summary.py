from pathlib import Path
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5TokenizerFast

from .model import NewsSummaryDataModule, NewsSummaryModel


def load_data(
    train_path: Path, test_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the news summary dataset.

    Args:
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.

    Returns:
        Tuple of training and test DataFrames with 'summary' and 'text' columns.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the dataset is empty or has invalid columns.
    """
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

    train_df = pd.read_csv(train_path, encoding="latin-1")
    test_df = pd.read_csv(test_path, encoding="latin-1")

    for df in (train_df, test_df):
        if df.empty:
            raise ValueError("Dataset is empty")
        if "text" not in df.columns or "ctext" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'ctext' columns")
        df = df[["text", "ctext"]]
        df.columns = ["summary", "text"]
        df.dropna(inplace=True)

    return train_df, test_df


def train_model(
    train_path: Path = Path("Project_NLP/data/processed/train_summary.csv"),
    test_path: Path = Path(
        "C:/Users/salma/OneDrive/Desktop/UNI/Spring 2025/DSAI 353/Project_NLP/data/processed/test_summary.csv"
    ),
    model_name: str = "t5-base",
    batch_size: int = 8,
    n_epochs: int = 3,
    checkpoint_dir: Path = Path("models/checkpoints"),
    log_dir: Path = Path("models/lightning_logs"),
) -> None:
    """Train the T5-based news summarization model.

    Args:
        train_path: Path to the training CSV file.
        test_path: Path to the test CSV file.
        model_name: Name of the pretrained T5 model.
        batch_size: Batch size for data loaders.
        n_epochs: Number of training epochs.
        checkpoint_dir: Directory to save model checkpoints.
        log_dir: Directory for TensorBoard logs.

    Raises:
        FileNotFoundError: If data files or directories are not found.
        RuntimeError: If training fails due to device or model issues.
    """
    # Ensure directories exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, test_df = load_data(train_path, test_path)

    # Initialize tokenizer and data module
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    data_module = NewsSummaryDataModule(
        train_df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    # Initialize model
    model = NewsSummaryModel(model_name=model_name)

    # Set up callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="validation_loss",
        mode="min",
    )
    logger = TensorBoardLogger(log_dir, name="news-summary")

    # Initialize trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
    )

    # Train the model
    try:
        trainer.fit(model, data_module)
    except RuntimeError as e:
        raise RuntimeError(f"Training failed: {e}")

    # Save the trained model
    model.model.save_pretrained(checkpoint_dir / "trained_model")
    tokenizer.save_pretrained(checkpoint_dir / "trained_model")


if __name__ == "__main__":
    pl.seed_everything(42)
    train_model()