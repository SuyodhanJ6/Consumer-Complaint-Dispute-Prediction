from pathlib import Path
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import (
#     Tokenizer,
#     HashingTF,
#     IDF,
#     VectorAssembler,
#     StandardScaler,
# )
# from consumerComplaint.config.spark_manager import spark_session
# from pyspark.sql import DataFrame
# import os
# import sys
# from consumerComplaint.ml.feature import FrequencyImputer, DerivedFeatureGenerator
# from consumerComplaint.exception import ConsumerComplaintException
# from consumerComplaint.logger import logger
# from consumerComplaint.entity.schema import FinanceDataSchema
# from consumerComplaint.entity.artifact_entity import DataTransformationArtifact
# from pyspark.ml.feature import StringIndexer, OneHotEncoder
# from pyspark.ml.feature import Imputer, FrequencyImputer

import os
import sys
import joblib
import dill
from dataclasses import dataclass
from typing import List
from pyspark.ml.feature import StandardScaler, VectorAssembler, OneHotEncoder, StringIndexer, Imputer
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, rand

from consumerComplaint.utils.main_utils import save_object
from consumerComplaint.config.spark_manager import spark_session
from consumerComplaint.entity.schema import FinanceDataSchema
from consumerComplaint.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from consumerComplaint.entity.config_entity import DataTransformationConfig
from consumerComplaint.exception import ConsumerComplaintException
from consumerComplaint.logger import logger
from consumerComplaint.ml.feature import FrequencyImputer, DerivedFeatureGenerator
from pyspark.ml.feature import IDF, Tokenizer, HashingTF

class DataTransformation:
    """
    Name: DataTransformation
    Description: This class performs data transformation operations.
    """
    def __init__(self, data_tf_config: DataTransformationConfig, data_val_artifact: DataValidationArtifact):
        """
        Name: __init__
        Description: Initializes the DataTransformation class.
        
        :param data_tf_config: An instance of DataTransformationConfig containing configuration settings.
        :param data_val_artifact: An instance of DataValidationArtifact containing data validation artifact.
        :return: None
        :raises: ConsumerComplaintException if an error occurs during initialization.
        :version: 1.0
        """
        try:
            self.data_tf_config = data_tf_config
            self.data_val_artifact = data_val_artifact
            self.schema = FinanceDataSchema()
        except Exception as e:
            raise ConsumerComplaintException(e, sys) from e


    def read_data(self) -> DataFrame:
        """
        Name: read_data
        Description: Reads the data from the accepted file path and returns it as a DataFrame.
        
        :return: A DataFrame containing the data.
        :raises: ConsumerComplaintException if an error occurs during reading.
        :version: 1.0
        """
        try:
            file_path = str(self.data_val_artifact.accepted_file_path)
            logger.info(f"Entering read_data method. Reading data from: {file_path}")
            dataframe: DataFrame = spark_session.read.parquet(file_path)
            dataframe.printSchema()
            logger.info("Data read successfully.")
            return dataframe
        except Exception as e:
            # logger.error(f"Error occurred in read_data method: {str(e)}")
            raise ConsumerComplaintException(e, sys) from e


    def get_data_transformation_pipeline(self) -> Pipeline:
        """
        Name: get_data_transformation_pipeline
        Description: Creates and returns a data transformation pipeline for feature engineering.

        This pipeline includes stages for numerical column transformation, adding derived features,
        imputing missing values, one-hot encoding, TF-IDF transformation, and more.

        :return: A PySpark Pipeline object representing the data transformation process.
        :raises: ConsumerComplaintException if an error occurs during pipeline creation.
        :version: 1.0
        """
        try:
            logger.info("Creating data transformation pipeline...")
            stages = []

            # Numerical column transformation

            # Generating additional columns
            derived_feature = DerivedFeatureGenerator(
                inputCols=self.schema.derived_input_features,
                outputCols=self.schema.derived_output_features
            )
            stages.append(derived_feature)
            logger.info("Added DerivedFeatureGenerator stage to the pipeline.")

            # Creating imputer to fill null values
            imputer = Imputer(
                inputCols=self.schema.numerical_columns,
                outputCols=self.schema.im_numerical_columns,
                strategy="mean"  # Modify the strategy as needed
            )
            stages.append(imputer)
            logger.info("Added Imputer stage to the pipeline.")

            frequency_imputer = FrequencyImputer(
                inputCols=self.schema.one_hot_encoding_features,
                outputCols=self.schema.im_one_hot_encoding_features
            )
            stages.append(frequency_imputer)
            logger.info("Added FrequencyImputer stage to the pipeline.")

            for im_one_hot_feature, string_indexer_col in zip(
                self.schema.im_one_hot_encoding_features,
                self.schema.string_indexer_one_hot_features
            ):
                string_indexer = StringIndexer(
                    inputCol=im_one_hot_feature,
                    outputCol=string_indexer_col
                )
                stages.append(string_indexer)
                logger.info(f"Added StringIndexer stage for {im_one_hot_feature} to the pipeline.")

            one_hot_encoder = OneHotEncoder(
                inputCols=self.schema.string_indexer_one_hot_features,
                outputCols=self.schema.tf_one_hot_encoding_features
            )
            stages.append(one_hot_encoder)
            logger.info("Added OneHotEncoder stage to the pipeline.")

            tokenizer = Tokenizer(
                inputCol=self.schema.tfidf_features[0],
                outputCol="words"
            )
            stages.append(tokenizer)
            logger.info("Added Tokenizer stage to the pipeline.")

            hashing_tf = HashingTF(
                inputCol=tokenizer.getOutputCol(),
                outputCol="rawFeatures",
                numFeatures=40
            )
            stages.append(hashing_tf)
            logger.info("Added HashingTF stage to the pipeline.")

            idf = IDF(
                inputCol=hashing_tf.getOutputCol(),
                outputCol=self.schema.tf_tfidf_features[0]
            )
            stages.append(idf)
            logger.info("Added IDF stage to the pipeline.")

            vector_assembler = VectorAssembler(
                inputCols=self.schema.input_features,
                outputCol=self.schema.vector_assembler_output
            )
            stages.append(vector_assembler)
            logger.info("Added VectorAssembler stage to the pipeline.")

            standard_scaler = StandardScaler(
                inputCol=self.schema.vector_assembler_output,
                outputCol=self.schema.scaled_vector_input_features
            )
            stages.append(standard_scaler)
            logger.info("Added StandardScaler stage to the pipeline.")

            pipeline = Pipeline(
                stages=stages
            )
            logger.info("Data transformation pipeline creation completed.")
            return pipeline

        except Exception as e:
            logger.error(f"Error while creating data transformation pipeline: {str(e)}")
            raise ConsumerComplaintException(e, sys)

    def save_pipeline(self, pipeline: Pipeline, export_dir: str):
        """
        Save a PySpark pipeline to the specified directory.

        :param pipeline: The PySpark pipeline to be saved.
        :param export_dir: The directory where the pipeline will be saved.
        :raises: ConsumerComplaintException if an error occurs during saving.
        """
        try:
            # Convert the Path object to a string
            export_dir_str = str(export_dir)
            pipeline.write().save(export_dir_str)
            logger.info(f"Pipeline saved to directory: {export_dir_str}")
        except Exception as e:
            logger.error(f"Error saving pipeline to {export_dir_str}: {str(e)}")
            # You can choose to raise an exception here or handle the error as needed.



    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Name: initiate_data_transformation
        Description: Initiates the data transformation process including feature engineering, train-test split,
                    and saving transformation artifacts.

        This method reads the input data, splits it into training and testing datasets, creates a data transformation
        pipeline, fits the pipeline on the training data, and then applies the transformation to both training and testing
        datasets. It also saves the transformation pipeline and the transformed data in specified directories.

        :return: A DataTransformationArtifact containing the file paths of transformed data and the exported pipeline.
        :raises: ConsumerComplaintException if any error occurs during the transformation process.
        :version: 1.0
        """
        
        try:
            logger.info("Started data transformation")
            dataframe: DataFrame = self.read_data()

            test_size = self.data_tf_config.test_size
            logger.info(f"Splitting dataset into train and test set using ratio: {1 - test_size}:{test_size}")
            train_dataframe, test_dataframe = dataframe.randomSplit([1 - test_size, test_size])

            logger.info(f"Train dataset has number of rows: [{train_dataframe.count()}] and "
                        f"columns: [{len(train_dataframe.columns)}]")

            # Create a data transformation pipeline
            pipeline = self.get_data_transformation_pipeline()

            # Fit the pipeline to the training data
            transformed_pipeline = pipeline.fit(train_dataframe)

            # Selecting required columns
            required_columns = [self.schema.scaled_vector_input_features, self.schema.target_column]

            # Transform the training and test data using the pipeline
            transformed_trained_dataframe = transformed_pipeline.transform(train_dataframe)
            transformed_trained_dataframe = transformed_trained_dataframe.select(required_columns)

            transformed_test_dataframe = transformed_pipeline.transform(test_dataframe)
            transformed_test_dataframe = transformed_test_dataframe.select(required_columns)

            export_pipeline_dir = Path(self.data_tf_config.export_pipeline_dir)
            transformed_train_dir = Path(self.data_tf_config.transformed_train_dir)
            transformed_test_dir = Path(self.data_tf_config.transformed_test_dir)

            # Create the required directories if they don't exist
            export_pipeline_dir.mkdir(parents=True, exist_ok=True)
            transformed_train_dir.mkdir(parents=True, exist_ok=True)
            transformed_test_dir.mkdir(parents=True, exist_ok=True)

            # Define the file paths
            transformed_train_data_file_path = transformed_train_dir / self.data_tf_config.file_name
            transformed_test_data_file_path = transformed_test_dir / self.data_tf_config.file_name

            logger.info(f"Saving transformation pipeline at: [{export_pipeline_dir}]")
            transformed_pipeline.save(str(export_pipeline_dir))
            # Specify the save mode as a string
            save_mode = "overwrite"  # You can change this to your desired mode: "error", "append", "overwrite", or "ignore"

            # Save the DataFrame
            # transformed_pipeline.write.mode(save_mode).save(str(export_pipeline_dir))


            # Save the DataFrame 
            # transformed_pipeline.write.mode(save_mode).save(str(export_pipeline_dir))
            

            # Define file paths for transformed data
            transformed_train_data_file_path = transformed_train_dir / self.data_tf_config.file_name
            transformed_test_data_file_path = transformed_test_dir / self.data_tf_config.file_name


            # Write transformed data to Parquet format
            transformed_trained_dataframe.write.parquet(str(transformed_train_data_file_path))
            transformed_test_dataframe.write.parquet(str(transformed_test_data_file_path))

            # Create a data transformation artifact
            data_tf_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_data_file_path,
                transformed_test_file_path=transformed_test_data_file_path,
                exported_pipeline_file_path=export_pipeline_dir,
            )

            logger.info(f"Data transformation artifact: {data_tf_artifact}")
            return data_tf_artifact
        except Exception as e:
            raise ConsumerComplaintException(e, sys)
