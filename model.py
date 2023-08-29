import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import AutoModel

# This model is bert based, fine tuned for uncased data

# You may notice there is no iterating over epochs, this is due to pytorch lightning
# handling that for us, we only need to define the steps for each epoch
# This is a huge advantage, as it saves us a lot of time and boilerplate

class BertModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        """Initialize BertModel from pretrained bert_uncased model"""
        super(BertModel, self).__init__()
        self.save_hyperparameters()  # Hereditary function from pl.LightningModule

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)  # Linear initialization
        self.num_classes = 2  # Binary classification for incorrect or correct

    def forward(self, input_ids, attention_mask):
        """Forward pass for BertModel
        This step feeds data into the next layer of a model
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step for BertModel
        This step gets feedback from the forward pass in the way of cross entropy
        (due to being binary classification) and logs it as train loss,
        we are training the model to minimize this loss, but not to much as to
        enter overfitting
        """
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for BertModel
        This step gets feedback from the forward pass in the way of cross entropy, again
        due to bbeing binary, but this time we are predicting the labels and comparing
        them to the actual labels, then we log the accuracy of the model, this is
        crucial to develop the layers correctly
        """
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
    
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
