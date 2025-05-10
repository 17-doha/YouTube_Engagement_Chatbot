from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AdamW


class NewsSummaryDataset(Dataset):
    """Dataset for news summary data compatible with T5 tokenizer."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5TokenizerFast,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128,
    ) -> None:
        """Initialize the dataset.

        Args:
            data: DataFrame containing text and summary columns.
            tokenizer: T5 tokenizer for encoding text.
            text_max_token_len: Maximum token length for input text.
            summary_max_token_len: Maximum token length for summaries.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample.

        Returns:
            Dictionary containing encoded text, summary, and attention masks.
        """
        data_row = self.data.iloc[index]
        text = data_row["text"]
        summary = data_row["summary"]

        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return {
            "text": text,
            "summary": summary,
            "text_input_ids": text_encoding["input_ids"].flatten(),
            "text_attention_mask": text_encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
            "labels_attention_mask": summary_encoding["attention_mask"].flatten(),
        }


class NewsSummaryDataModule(pl.LightningDataModule):
    """Data module for news summary dataset."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5TokenizerFast,
        batch_size: int = 8,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128,
    ) -> None:
        """Initialize the data module.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            tokenizer: T5 tokenizer.
            batch_size: Batch size for data loaders.
            text_max_token_len: Maximum token length for input text.
            summary_max_token_len: Maximum token length for summaries.
        """
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and testing.

        Args:
            stage: Optional stage name (e.g., 'fit', 'test').
        """
        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len,
        )
        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )


class NewsSummaryModel(pl.LightningModule):
    """T5-based model for news summarization."""

    def __init__(self, model_name: str = "t5-base") -> None:
        """Initialize the model.

        Args:
            model_name: Name of the pretrained T5 model.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, return_dict=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask for input.
            decoder_attention_mask: Attention mask for decoder.
            labels: Target labels for training.

        Returns:
            Tuple of loss and logits.
        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return output.loss, output.logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        loss, _ = self(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            decoder_attention_mask=batch["labels_attention_mask"],
            labels=batch["labels"],
        )
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        loss, _ = self(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            decoder_attention_mask=batch["labels_attention_mask"],
            labels=batch["labels"],
        )
        self.log("validation_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a test step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        loss, _ = self(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            decoder_attention_mask=batch["labels_attention_mask"],
            labels=batch["labels"],
        )
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> AdamW:
        """Configure the optimizer.

        Returns:
            AdamW optimizer.
        """
        return AdamW(self.parameters(), lr=0.0001)


def summarize_text(
    text: str,
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    max_length: int = 128,
    num_beams: int = 4,
    device: str = "cuda:0",
) -> str:
    """Generate a summary for the input text using the T5 model.

    Args:
        text: Input text to summarize.
        model: T5 model for generation.
        tokenizer: T5 tokenizer for encoding.
        max_length: Maximum length of the generated summary.
        num_beams: Number of beams for beam search.
        device: Device to run the model on (e.g., 'cuda:0' or 'cpu').

    Returns:
        Generated summary.

    Raises:
        RuntimeError: If the device is unavailable or model fails to generate.
    """
    try:
        inputs = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to generate summary: {e}")