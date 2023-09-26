from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    metadata_file_path: str
    download_dir: str


@dataclass
class DataValidationArtifact:
    accepted_file_path: Path
    rejected_dir: Path


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: Path
    exported_pipeline_file_path: Path
    transformed_test_file_path: Path

