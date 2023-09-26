from consumerComplaint.config.spark_manager import spark_session

# da = spark_session.read.parquet("/home/suyodhan/Documents/Data-Science-Project/Consumer-Complaint-Dispute-Prediction/consumer_artifact/data_validation/20230926_003644/accepted_data/consumer_complaint")
from pyspark.sql import DataFrame


file_path  = "/home/suyodhan/Documents/Data-Science-Project/Consumer-Complaint-Dispute-Prediction/consumer_artifact/data_validation/20230926_005416/accepted_data/consumer_complaint"

dataframe: DataFrame = spark_session.read.parquet(file_path)
# dataframe.printSchema()
dataframe.show(10)