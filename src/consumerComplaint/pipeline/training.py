import sys

from consumerComplaint.exception import ConsumerComplaintException
from consumerComplaint.logger import logger
from consumerComplaint.config.pipeline.training import FinanceConfig
from consumerComplaint.components.training.data_ingestion import DataIngestion
from consumerComplaint.components.training.data_validation import DataValidation
from consumerComplaint.components.training.data_transformation import DataTransformation
from consumerComplaint.components.training.model_training import ModelTrainer
from consumerComplaint.components.training.model_evalution import ModelEvaluation
from consumerComplaint.components.training.model_pusher import ModelPusher
from consumerComplaint.entity.artifact_entity import (DataIngestionArtifact, 
                                                      DataValidationArtifact, 
                                                      DataTransformationArtifact, 
                                                      ModelTrainerArtifact, 
                                                      ModelEvaluationArtifact)


class TrainingPipeline:

    def __init__(self, finance_config: FinanceConfig):
        self.finance_config: FinanceConfig = finance_config

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.finance_config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact

        except Exception as e:
            raise ConsumerComplaintException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = self.finance_config.get_data_validation_config()
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)

            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise ConsumerComplaintException(e, sys)

    def start_data_transformation(self, data_val_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = self.finance_config.get_data_transformation_config()
            data_transformation = DataTransformation(data_val_artifact=data_val_artifact,
                                                     data_tf_config=data_transformation_config

                                                     )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise ConsumerComplaintException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = self.finance_config.get_model_trainer_config()
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=model_trainer_config
                                         )
            model_trainer_artifact = model_trainer.initiate_model_training()
            return model_trainer_artifact
        except Exception as e:
            raise ConsumerComplaintException(e, sys)

    def start_model_evaluation(self, data_validation_artifact, model_trainer_artifact) -> ModelEvaluationArtifact:
        try:
            model_eval_config = self.finance_config.get_model_evaluation_config()
            model_eval = ModelEvaluation(data_validation_artifact=data_validation_artifact,
                                         model_trainer_artifact=model_trainer_artifact,
                                         model_eval_config=model_eval_config
                                         )
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise ConsumerComplaintException(e, sys)

    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact):
        try:
            model_pusher_config = self.finance_config.get_model_pusher_config()
            model_pusher = ModelPusher(model_trainer_artifact=model_trainer_artifact,
                                       model_pusher_config=model_pusher_config
                                       )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise ConsumerComplaintException(e, sys)

    def start(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_val_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact=data_validation_artifact,
                                                              model_trainer_artifact=model_trainer_artifact
                                                              )
            if model_eval_artifact.model_accepted:
                self.start_model_pusher(model_trainer_artifact=model_trainer_artifact)
        except Exception as e:
            raise ConsumerComplaintException(e, sys)

