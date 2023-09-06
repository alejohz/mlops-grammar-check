from fastapi import FastAPI

from inference import ColaONNXPredictor

app = FastAPI(title="BERT Binary Classification API")

predictor = ColaONNXPredictor("./models/model.onnx")


@app.get("/")
async def home():
    return "Binary Classification with the Bert model"


@app.get("/predict/")
async def get_prediction(text: str):
    prediction = predictor.predict(text)
    return str(prediction[0])
