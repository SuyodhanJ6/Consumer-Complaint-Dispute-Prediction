import os, sys
import json

from consumerComplaint.pipeline.training import TrainingPipeline
from consumerComplaint.pipeline.prediction import PredictionPipeline
from consumerComplaint.config.pipeline.training import FinanceConfig
from consumerComplaint.constants.applicaton import APP_HOST, APP_PORT
from consumerComplaint.pipeline.local_predict import make_predictions
from consumerComplaint.config.spark_manager import spark_session
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
            "Training successful and predictions are stored in s3 bucket !!"
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@app.get("/predict")
async def predict_route():
    try:
        df_file_path = "/home/suyodhan/Documents/Data-Science-Project/Consumer-Complaint-Dispute-Prediction/predictoin_data"

        df = spark_session.read.parquet(df_file_path)

        # Assuming you have 'df' as your input DataFrame
        predictions_df = make_predictions(df)

        # Convert the predictions DataFrame to JSON
        predictions_json = predictions_df.toJSON().collect()

        # Convert the JSON data to a list of dictionaries
        predictions_list = [json.loads(row) for row in predictions_json]

        # Return the predictions as JSON in the response
        return {"predictions": predictions_list}

        # prediction = PredictionPipeline()
        # prediction.start_batch_prediction()

    except Exception as e:
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    run_app(app=app, host= APP_HOST, port= APP_PORT)
    