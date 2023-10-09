from consumerComplaint.entity.estimator import S3FinanceEstimator
from consumerComplaint.constants.model import S3_MODEL_BUCKET_NAME, S3_MODEL_DIR_KEY
from pyspark.sql import DataFrame

def make_predictions(input_df: DataFrame) -> DataFrame:
    # Initialize the S3FinanceEstimator with the appropriate bucket and key
    estimator = S3FinanceEstimator(bucket_name=S3_MODEL_BUCKET_NAME, s3_key=S3_MODEL_DIR_KEY)

    # Use the estimator to make predictions on the input DataFrame
    predictions = estimator.transform(input_df)

    return predictions