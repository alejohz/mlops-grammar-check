import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# This project uses pytorch lighting to avoid some pytorch boilerplate


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=64,
        max_length=128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, dtype=torch.float32)

    def prepare_data(self):
        # COLA means Corpus of Linguistic Acceptability
        cola_dataset = load_dataset("glue", "cola")  # Load dataset
        self.train_data = cola_dataset["train"]  # Split into train and validation
        self.val_data = cola_dataset["validation"]  # This split comes from hf datasets

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Stage check to avoid loading unnecesary data
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label", "sentence"],
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label", "sentence"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    # Add manual data validation
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    # Print to check shape
    # Shape is very important when using pytorch
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
    # torch.Size([32, 256])
    # This is the size expected due to model batch being 32 and max_length being 256
