import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data_load import DataModule
from model import BertModel

load_dotenv()
# Adding env variable TOKENIZERS_PARALLELISM and setting as false to avoid deadlocks
# when forking processes
torch.set_default_dtype(torch.float32)
# You may be wodering why we are training a model if inside model.py
# it says from pre trained or already trained. This is because we are
# fine tuning the model to our data, this is a very important step
# as it allows us to use the model for our specific use case, in this
# case, the model is being fine tuned to the HF COLA dataset.


def main():
    """Main function to train model"""
    cola_data = DataModule()
    bert_model = BertModel()
    # Model checkpoint is a callback that allows us to save the
    # model after each epoch, this is very useful as we can use
    # the model that performed the best on the validation set.
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    # Early stop is highly recommended, specially when fine tunning.
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    # Here we can see some really pytorch lightning magic, we are
    # using the trainer class to train our model, we are passing
    # some parameters to the trainer class, such as the number of
    # epochs, the logger, the callbacks, etc. This is a very
    # important step as it allows us to train our model in a very
    # easy way, we don't have to worry about the training loop,
    # the validation loop, the test loop, etc. Pytorch lightning
    # does all of that for us, we just have to pass the parameters
    # and it will do the rest.
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator="cpu",
        # Had to use CPU as metal has no support for float64
        max_epochs=5,  # Epochs not always mean better results, but 5 is low, normally
        # 20 is a good number to start with. This is a demo
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(bert_model, cola_data)


if __name__ == "__main__":
    main()
