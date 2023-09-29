from dataclasses import dataclass
from pathlib import Path
from consumerComplaint.constants.prediction_pipeline_config.file_config import ARCHIVE_DIR, INPUT_DIR, FAILED_DIR, \
    PREDICTION_DIR, REGION_NAME

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str
    artifact_dir: str


@dataclass
class DataIngestionConfig:
    from_date : str
    to_date: int
    data_ingestion_dir : Path
    download_dir : Path
    file_name: str
    feature_store_dir: Path
    failed_dir:Path
    metadata_file_path: Path
    datasource_url: str


@dataclass
class DataValidationConfig:
    accepted_data_dir: Path
    rejected_data_dir: Path
    file_name: str


@dataclass 
class DataTransformationConfig:
    file_name: str
    export_pipeline_dir: Path
    transformed_train_dir: Path
    transformed_test_dir: Path
    test_size: float


@dataclass
class ModelTrainerConfig:
    base_accuracy: float
    trained_model_file_path: str
    metric_list: list
    label_indexer_model_dir: str



@dataclass
class ModelEvaluationConfig:
    model_evaluation_report_file_path: str
    threshold: float
    metric_list: list
    model_dir: str
    bucket_name: str

@dataclass
class ModelPusherConfig:
    model_dir: str
    bucket_name: str


class PredictionPipelineConfig:

    def __init__(self, input_dir=INPUT_DIR,
                 prediction_dir=PREDICTION_DIR,
                 failed_dir=FAILED_DIR,
                 archive_dir=ARCHIVE_DIR,
                 region_name=REGION_NAME
                 ):
        self.input_dir = input_dir
        self.prediction_dir = prediction_dir
        self.failed_dir = failed_dir
        self.archive_dir = archive_dir
        self.region_name = region_name

    def to_dict(self):
        return self.__dict__
