import torch

from data_load import DataModule
from model import BertModel


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


if __name__ == "__main__":
    sentences = [
        "The boys is sitting on the bench",  # unacceptable
        "A boy is sitting alone on the bench",  # acceptable
    ]
    for sentence in sentences:
        predictor = ColaPredictor("./models/epoch=2-step=804.ckpt")
        print(predictor.predict(sentence))
