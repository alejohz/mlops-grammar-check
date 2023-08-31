import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from transformers import AutoModel

import wandb

# This model is bert based, fine tuned for uncased data

# You may notice there is no iterating over epochs, this is due to pytorch lightning
# handling that for us, we only need to define the steps for each epoch
# This is a huge advantage, as it saves us a lot of time and boilerplate


class BertModel(pl.LightningModule):
    def __init__(
        self, model_name: str = "google/bert_uncased_L-2_H-128_A-2", lr: float = 1e-7
    ):
        """Initialize BertModel from pretrained bert_uncased model"""
        super(BertModel, self).__init__()
        self.save_hyperparameters()  # Hereditary function from pl.LightningModule

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)  # Linear initialization
        self.num_classes = 2  # Binary classification for incorrect or correct
        self.task = "binary"
        self.train_accuracy_metric = torchmetrics.Accuracy(task=self.task)
        self.val_accuracy_metric = torchmetrics.Accuracy(task=self.task)
        self.f1_metric = torchmetrics.F1Score(
            num_classes=self.num_classes, task=self.task
        )
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes, task=self.task
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes, task=self.task
        )
        self.precision_micro_metric = torchmetrics.Precision(
            average="micro", task=self.task
        )
        self.recall_micro_metric = torchmetrics.Recall(average="micro", task=self.task)

    def forward(
        self, input_ids: torch.tensor, attention_mask: torch.tensor, labels=torch.tensor
    ):
        """Forward pass for BertModel
        This step feeds data into the next layer of a model
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        return outputs

    def training_step(self, batch: torch.Tensor, batch_idx):
        """Training step for BertModel
        This step gets feedback from the forward pass in the way of cross entropy
        (due to being binary classification) and logs it as train loss,
        we are training the model to minimize this loss, but not to much as to
        enter overfitting
        """
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        """Validation step for BertModel
        This step gets feedback from the forward pass in the way of cross entropy, again
        due to bbeing binary, but this time we are predicting the labels and comparing
        them to the actual labels, then we log the accuracy of the model, this is
        crucial to develop the layers correctly
        """
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        self.log("valid/f1", f1, prog_bar=True)
        return {"labels": labels, "logits": outputs.logits}

    def on_validation_epoch_end(self, outputs):
        """Validation epoch end for BertModel
        We are using a confusion matrix to see how the model is performing on each
        class, and if there is any class imbalance, we can see it here.
        """
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)
        wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

    # Here we are skipping the test step, as it is not necessary for this project
    # but for a production model, it is desirable to have a test step
    # Normally, test step sees model drift (If there is any) faster, due to unseen data

    def configure_optimizers(self):
        """Configure optimizers for BertModel
        This step is crucial to train the model, we are using Adam as the optimizer
        and the learning rate is set to 1e-2, this is standard for Adam,
        but it can be changed to SGD or any other optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
