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