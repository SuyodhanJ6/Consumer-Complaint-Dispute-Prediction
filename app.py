import os, sys


from consumerComplaint.pipeline.training import TrainingPipeline
from consumerComplaint.pipeline.prediction import PredictionPipeline
from consumerComplaint.config.pipeline.training import FinanceConfig
from consumerComplaint.constants.applicaton import APP_HOST, APP_PORT


from fastapi import FastAPI
from fastapi.responses import Response
from uvicorn import run as run_app



app = FastAPI()

@app.get("/train")
async def train_route():
    try:

        
        finance = FinanceConfig()
        training_pipeline = TrainingPipeline(finance)
        training_pipeline.start()

        return Response(
            "Prediction successful and predictions are stored in s3 bucket !!"
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@app.get("/predict")
async def predict_route():
    try:
        prediction = PredictionPipeline()
        prediction.start_batch_prediction()

    except Exception as e:
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    run_app(app=app, host= APP_HOST, port= APP_PORT)
    