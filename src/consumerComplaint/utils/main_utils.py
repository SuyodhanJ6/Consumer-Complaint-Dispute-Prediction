import os
import sys
import shutil
import dill
import yaml
from typing import List
from pyspark.sql import DataFrame
from consumerComplaint.logger import logger
from consumerComplaint.exception import ConsumerComplaintException
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def write_yaml_file(file_path: str, data: dict = None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise ConsumerComplaintException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ConsumerComplaintException(e, sys) from e


def get_score(dataframe: DataFrame, metric_name, label_col, prediction_col) -> float:
    try:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=prediction_col,
            metricName=metric_name)
        score = evaluator.evaluate(dataframe)
        print(f"{metric_name} score: {score}")
        logger.info(f"{metric_name} score: {score}")
        return score
    except Exception as e:
        raise ConsumerComplaintException(e, sys)


def create_directories(directories_list: List[str], new_directory=False):
    try:

        for dir_path in directories_list:
            if dir_path.startswith("s3"):
                continue
            if os.path.exists(dir_path) and new_directory:
                shutil.rmtree(dir_path)
                logger.info(f"Directory removed: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory created: {dir_path}")
    except Exception as e:
        raise ConsumerComplaintException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info("Entered the save_object method of MainUtils class")
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise ConsumerComplaintException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ConsumerComplaintException(e, sys) from e