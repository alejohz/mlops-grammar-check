import numpy as np
import onnxruntime as ort
import torch
from scipy.special import softmax

from data_load import DataModule
from model import BertModel
from utils import timing


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


if __name__ == "__main__":
    sentences = [
        "The boys is sitting on the bench",  # unacceptable
        "A boy is sitting alone on the bench",  # acceptable
    ]
    for sentence in sentences:
        predictor = ColaPredictor("./models/best-checkpoint-v1.ckpt")
        print(predictor.predict(sentence))

    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
    for sentence in sentences:
        predictor.predict(sentence)
