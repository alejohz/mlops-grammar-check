import logging
import os
import hydra
import torch
from omegaconf.omegaconf import DictConfig

from data_load import DataModule
from model import BertModel

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def convert_model(cfg: DictConfig):
    """Convert the model into ONNX format."""
    os.environ["TOKENIZERS_PARALLELISM"] = str(cfg.model.parallelism)
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    bert_model = BertModel.load_from_checkpoint(model_path)

    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    # Export the model
    logger.info("Converting the model into ONNX format")
    torch.onnx.export(
        bert_model,  # model being run
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  
        # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        "Model converted successfully. ONNX format model is at:"
        f" {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()
