from dataclasses import dataclass
from pathlib import Path

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