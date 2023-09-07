import numpy as np
import onnxruntime as ort
import torch
from scipy.special import softmax

from data_load import DataModule
from model import BertModel
from utils import timing
import wandb
import hydra
from omegaconf import DictConfig


class ColaPredictor:
    def __init__(self, model_path):
        """Initialize the model and tokenizer."""
        self.model_path = model_path
        self.model = BertModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        """Predict the label of the text."""
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.labels = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


@hydra.main(config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sentences = [
        "The boys is sitting on the bench",  # unacceptable
        "A boy is sitting alone on the bench",  # acceptable
    ] * cfg.inference.sentences_multiplier
    wandb.init(
        project=cfg.wandb.project_name,
        entity=cfg.wandb.entity,
        job_type=cfg.inference.job_type,
        tags=cfg.wandb.tags,

    )
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    onnx_predictor = ColaONNXPredictor("./models/model.onnx")
    for sentence in sentences:
        prediction, time = predictor.predict(sentence)
        wandb.log(
            {
                "prediction": prediction[0]["label"],
                "score": prediction[0]["score"],
                "sentence": sentence,
                "time": time,
                "type": "ckpt",
            }
        )
        onnx_prediction, onnx_time = onnx_predictor.predict(sentence)
        wandb.log(
            {
                "prediction": onnx_prediction[0]["label"],
                "score": onnx_prediction[0]["score"],
                "sentence": sentence,
                "time": onnx_time,
                "type": "onnx",
            }
        )
    wandb.finish()


if __name__ == "__main__":
    main()
