# import joblib

# # Sample data to serialize
# data_to_serialize = {'name': 'John', 'age': 30, 'city': 'New York'}

# # Serialize the data to a file
# joblib.dump(data_to_serialize, 'data.pkl')

# # Deserialize the data from the file
# loaded_data = joblib.load('data.pkl')

# # Print the loaded data
# print(loaded_data)
from pyspark.ml.pipeline import Pipeline, PipelineModel

try:

    transformed_pipeline_file_path = "consumer_artifact/data_transformation/20230927_130037/transformed_pipeline/transformed_pipeline.joblib"
    transformed_pipeline = PipelineModel.load(transformed_pipeline_file_path)

except Exception as e:
    print(f"Error : {e}")