import sys
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, FloatType, StructType, StructField

from consumerComplaint.config.spark_manager import spark_session
from consumerComplaint.entity.artifact_entity import (
    ModelEvaluationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)
from consumerComplaint.entity.config_entity import ModelEvaluationConfig
from consumerComplaint.entity.estimator import S3FinanceEstimator
from consumerComplaint.entity.schema import FinanceDataSchema
from consumerComplaint.exception import ConsumerComplaintException
from consumerComplaint.logger import logger
from consumerComplaint.utils.main_utils import get_score
from consumerComplaint.data_access.model_eval_artifact import ModelEvaluationArtifactData



class ModelEvaluation:

    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 model_eval_config: ModelEvaluationConfig,
                 schema=FinanceDataSchema()
                 ):
        """
        Name: __init__
        Description: Initializes the ModelEvaluation class.

        This constructor initializes the ModelEvaluation class with data validation artifact, model trainer artifact,
        model evaluation configuration, schema, and other necessary attributes.

        :param data_validation_artifact: DataValidationArtifact containing information about accepted data.
        :param model_trainer_artifact: ModelTrainerArtifact containing information about the trained model.
        :param model_eval_config: ModelEvaluationConfig containing model evaluation configuration.
        :param schema: FinanceDataSchema containing schema information.
        :version: 1.0
        """
        try:
            self.model_eval_artifact_data = ModelEvaluationArtifactData()
            self.data_validation_artifact = data_validation_artifact
            self.model_eval_config = model_eval_config
            self.model_trainer_artifact = model_trainer_artifact
            self.schema = schema
            self.bucket_name = self.model_eval_config.bucket_name
            self.s3_model_dir_key = self.model_eval_config.model_dir
            self.s3_finance_estimator = S3FinanceEstimator(
                bucket_name=self.bucket_name,
                s3_key=self.s3_model_dir_key
            )
            self.metric_report_schema = StructType([StructField("model_accepted", StringType()),
                                                    StructField("changed_accuracy", FloatType()),
                                                    StructField("trained_model_path", StringType()),
                                                    StructField("best_model_path", StringType()),
                                                    StructField("active", StringType())]
                                                )

        except Exception as e:
            raise ConsumerComplaintException(e, sys)


    def read_data(self) -> DataFrame:
        """
        Name: read_data
        Description: Reads the accepted data for model evaluation.

        This method reads the accepted data for model evaluation from the data validation artifact.

        :return: A PySpark DataFrame containing the accepted data.
        :raises: ConsumerComplaintException if an error occurs during data reading.
        :version: 1.0
        """
        try:
            file_path = self.data_validation_artifact.accepted_file_path
            dataframe: DataFrame = spark_session.read.parquet(file_path)
            return dataframe
        except Exception as e:
            # Raising an exception.
            raise ConsumerComplaintException(e, sys)


    def evaluate_trained_model(self) -> ModelEvaluationArtifact:
        """
        Name: evaluate_trained_model
        Description: Evaluates the trained model's performance.

        This method evaluates the performance of the trained model by comparing it to the best available model.
        It calculates the F1 score for both the trained model and the best model and checks if the trained model's
        accuracy has improved beyond the configured threshold.

        :return: A ModelEvaluationArtifact containing information about model acceptance and performance changes.
        :version: 1.0
        """
        is_model_accepted, is_active = False, False
        trained_model_file_path = self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
        label_indexer_model_path = self.model_trainer_artifact.model_trainer_ref_artifact.label_indexer_model_file_path

        label_indexer_model = StringIndexerModel.load(label_indexer_model_path)
        trained_model = PipelineModel.load(trained_model_file_path)

        dataframe: DataFrame = self.read_data()
        dataframe = label_indexer_model.transform(dataframe)

        best_model_path = self.s3_finance_estimator.get_latest_model_path()
        trained_model_dataframe = trained_model.transform(dataframe)
        best_model_dataframe = self.s3_finance_estimator.transform(dataframe)

        trained_model_f1_score = get_score(dataframe=trained_model_dataframe, metric_name="f1",
                                        label_col=self.schema.target_indexed_label,
                                        prediction_col=self.schema.prediction_column_name)
        best_model_f1_score = get_score(dataframe=best_model_dataframe, metric_name="f1",
                                        label_col=self.schema.target_indexed_label,
                                        prediction_col=self.schema.prediction_column_name)

        logger.info(f"Trained_model_f1_score: {trained_model_f1_score}, Best model f1 score: {best_model_f1_score}")
        changed_accuracy = trained_model_f1_score - best_model_f1_score

        if changed_accuracy >= self.model_eval_config.threshold:
            is_model_accepted, is_active = True, True

        model_evaluation_artifact = ModelEvaluationArtifact(model_accepted=is_model_accepted,
                                                            changed_accuracy=changed_accuracy,
                                                            trained_model_path=trained_model_file_path,
                                                            best_model_path=best_model_path,
                                                            active=is_active
                                                            )
        return model_evaluation_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Name: initiate_model_evaluation
        Description: Initiates the model evaluation process.

        This method initiates the model evaluation process by checking if a trained model is available.
        If a trained model is available, it evaluates the model's performance. Otherwise, it assumes the trained
        model is accepted with zero accuracy change.

        :return: A ModelEvaluationArtifact containing information about model acceptance and performance changes.
        :version: 1.0
        """
        try:
            model_accepted = True
            is_active = True

            if not self.s3_finance_estimator.is_model_available(key=self.s3_finance_estimator.s3_key):
                latest_model_path = None
                trained_model_path = self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
                model_evaluation_artifact = ModelEvaluationArtifact(model_accepted=model_accepted,
                                                                    changed_accuracy=0.0,
                                                                    trained_model_path=trained_model_path,
                                                                    best_model_path=latest_model_path,
                                                                    active=is_active
                                                                    )
            else:
                model_evaluation_artifact = self.evaluate_trained_model()

            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            self.model_eval_artifact_data.save_eval_artifact(model_eval_artifact=model_evaluation_artifact)
            return model_evaluation_artifact
        except Exception as e:
            raise ConsumerComplaintException(e, sys)


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            model_accepted = True
            is_active = True

            if not self.s3_finance_estimator.is_model_available(key=self.s3_finance_estimator.s3_key):
                latest_model_path = None
                trained_model_path = self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
                model_evaluation_artifact = ModelEvaluationArtifact(model_accepted=model_accepted,
                                                                    changed_accuracy=0.0,
                                                                    trained_model_path=trained_model_path,
                                                                    best_model_path=latest_model_path,
                                                                    active=is_active
                                                                    )
            else:
                model_evaluation_artifact = self.evaluate_trained_model()

            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            self.model_eval_artifact_data.save_eval_artifact(model_eval_artifact=model_evaluation_artifact)
            return model_evaluation_artifact
        except Exception as e:
            raise ConsumerComplaintException(e, sys)
