import os
from pathlib import Path

PIPELINE_NAME = "consumer-complaint"
PIPELINE_ARTIFACT_DIR = Path.cwd() / "consumer_artifact"


# PIPELINE_ARTIFACT_DIR = os.path.join(os.getcwd(), "finance_artifact")

from consumerComplaint.constants.training_pipeline_config.data_ingestion import *
from consumerComplaint.constants.training_pipeline_config.data_validation import *
from consumerComplaint.constants.training_pipeline_config.data_transformation import *
from consumerComplaint.constants.training_pipeline_config.model_training import *
from consumerComplaint.constants.training_pipeline_config.model_evalution import *
from consumerComplaint.constants.training_pipeline_config.model_pusher import *
