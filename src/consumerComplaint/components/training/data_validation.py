import os
import sys
from collections import namedtuple
from typing import List, Dict
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit

from consumerComplaint.config.spark_manager import spark_session
from consumerComplaint.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from consumerComplaint.entity.config_entity import DataValidationConfig
from consumerComplaint.entity.schema import FinanceDataSchema
from consumerComplaint.exception import ConsumerComplaintException
from consumerComplaint.logger import logger

COMPLAINT_TABLE = "complaint"
ERROR_MESSAGE = "error_msg"
MissingReport = namedtuple("MissingReport", ["total_row", "missing_row", "missing_percentage"])


class DataValidation(FinanceDataSchema):
    """
    Name: DataValidation
    Description: This class handles the data validation process for Consumer Complaint data.
    """

    def __init__(self,
             data_validation_config: DataValidationConfig,
             data_ingestion_artifact: DataIngestionArtifact,
             table_name: str = COMPLAINT_TABLE,
             schema=FinanceDataSchema()
             ):
        """
        Name: __init__
        Description: Initializes a DataValidation instance.

        This constructor initializes a DataValidation instance with the provided data validation configuration,
        data ingestion artifact, table name, and schema.

        :param data_validation_config: A DataValidationConfig object containing configuration settings.
        :param data_ingestion_artifact: A DataIngestionArtifact object containing data ingestion artifacts.
        :param table_name: The name of the table to be validated (default is COMPLAINT_TABLE).
        :param schema: An optional schema object defining the data schema (default is FinanceDataSchema).
        :raises: ConsumerComplaintException if an error occurs during initialization.
        :version: 1.0
        """
        try:
            super().__init__()
            self.data_ingestion_artifact: DataIngestionArtifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.table_name = table_name
            self.schema = schema
        except Exception as e:
            raise ConsumerComplaintException(e, sys) from e
        
    def read_data(self) -> DataFrame:
        """
        Name: read_data
        Description: Reads and retrieves a DataFrame from the feature store file.

        This method reads and retrieves a DataFrame from the feature store file path specified in the
        data ingestion artifact. It limits the DataFrame to 10,000 rows for quick data exploration.

        :return: A PySpark DataFrame containing the data.
        :raises: ConsumerComplaintException if an error occurs during data reading.
        :version: 1.0
        """
        try:
            dataframe: DataFrame = spark_session.read.parquet(
                self.data_ingestion_artifact.feature_store_file_path
            ).limit(10000)
            logger.info(f"Data frame is created using file: {self.data_ingestion_artifact.feature_store_file_path}")
            logger.info(f"Number of rows: {dataframe.count()} and columns: {len(dataframe.columns)}")
            # Uncomment the line below if you want to sample a smaller portion of the data
            # dataframe, _ = dataframe.randomSplit([0.001, 0.999])
            return dataframe
        except Exception as e:
            raise ConsumerComplaintException(e, sys) from e


    @staticmethod
    def get_missing_report(dataframe: DataFrame, ) -> Dict[str, MissingReport]:
        try:
            missing_report: Dict[str:MissingReport] = dict()
            logger.info(f"Preparing missing reports for each column")
            number_of_row = dataframe.count()

            for column in dataframe.columns:
                missing_row = dataframe.filter(f"{column} is null").count()
                missing_percentage = (missing_row * 100) / number_of_row
                missing_report[column] = MissingReport(total_row=number_of_row,
                                                       missing_row=missing_row,
                                                       missing_percentage=missing_percentage
                                                       )
            logger.info(f"Missing report prepared: {missing_report}")
            return missing_report

        except Exception as e:
            raise ConsumerComplaintException(e, sys)

    def get_unwanted_and_high_missing_value_columns(self, dataframe: DataFrame, threshold: float = 0.2) -> List[str]:
        try:
            missing_report: Dict[str, MissingReport] = self.get_missing_report(dataframe=dataframe)

            unwanted_column: List[str] = self.schema.unwanted_columns
            for column in missing_report:
                if missing_report[column].missing_percentage > (threshold * 100):
                    unwanted_column.append(column)
                    logger.info(f"Missing report {column}: [{missing_report[column]}]")
            unwanted_column = list(set(unwanted_column))
            return unwanted_column

        except Exception as e:
            raise ConsumerComplaintException(e, sys)
        

    def drop_unwanted_columns(self, dataframe: DataFrame) -> DataFrame:
        """
        Name: drop_unwanted_columns
        Description: Drops unwanted columns and saves them to a separate file.

        This method identifies and drops columns that are unwanted or contain a high number of missing values.
        The dropped columns are also saved to a separate Parquet file for further analysis.

        :param dataframe: The input PySpark DataFrame.
        :return: A new PySpark DataFrame with unwanted columns removed.
        :raises: ConsumerComplaintException if an error occurs during column dropping or file writing.
        :version: 1.0
        """
        try:
            unwanted_columns: List = self.get_unwanted_and_high_missing_value_columns(dataframe=dataframe, )
            logger.info(f"Dropping feature: {','.join(unwanted_columns)}")
            unwanted_dataframe: DataFrame = dataframe.select(unwanted_columns)

            unwanted_dataframe = unwanted_dataframe.withColumn(ERROR_MESSAGE, lit("Contains many missing values"))

            rejected_dir = Path(self.data_validation_config.rejected_data_dir) / "missing_data"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            file_path = Path(rejected_dir) / self.data_validation_config.file_name

            logger.info(f"Writing dropped columns into file: [{file_path}]")

            # Save the dropped columns to a Parquet file
            unwanted_dataframe.write.mode("append").parquet(str(file_path))

            dataframe: DataFrame = dataframe.drop(*unwanted_columns)
            logger.info(f"Remaining number of columns: [{len(dataframe.columns)}]")
            return dataframe
        except Exception as e:
            raise ConsumerComplaintException(e, sys) from e


    @staticmethod
    def get_unique_values_of_each_column(dataframe: DataFrame) -> None:
        """
        Name: get_unique_values_of_each_column
        Description: Calculates and logs the unique values, missing values, and missing percentage for each column.

        This static method takes a PySpark DataFrame as input, iterates through its columns, and calculates the number of
        unique values, missing values, and missing percentage for each column. The results are logged for analysis.

        :param dataframe: The input PySpark DataFrame.
        :return: None
        :raises: ConsumerComplaintException if an error occurs during value calculation.
        :version: 1.0
        """
        try:
            for column in dataframe.columns:
                n_unique: int = dataframe.select(col(column)).distinct().count()
                n_missing: int = dataframe.filter(col(column).isNull()).count()
                missing_percentage: float = (n_missing * 100) / dataframe.count()
                logger.info(f"Column: {column} contains {n_unique} unique values and missing percentage: {missing_percentage} %.")
        except Exception as e:
            raise ConsumerComplaintException(e, sys)


    def is_required_columns_exist(self, dataframe: DataFrame):
        """
        Name: is_required_columns_exist
        Description: Checks if all required columns exist in the DataFrame.

        This method verifies whether all the required columns, as defined in the schema, exist in the input DataFrame.
        If any required column is missing, it raises an exception with details of the expected and found columns.

        :param dataframe: The input PySpark DataFrame.
        :return: None
        :raises: ConsumerComplaintException if required columns are missing.
        :version: 1.0
        """
        try:
            columns = list(filter(lambda x: x in self.schema.required_columns, dataframe.columns))
            logger.info(f"1, Columns Count in is_required function: {len(columns)}")
            logger.info(f"2, Columns Count in is_required function: {len(self.schema.required_columns)}")
            if len(columns) != len(self.schema.required_columns):
                raise Exception(f"Required column(s) missing\n\
                    Expected columns: {self.schema.required_columns}\n\
                    Found columns: {columns}\
                    ")
        except Exception as e:
            raise ConsumerComplaintException(e, sys)


    # def drop_row_without_target_label(self, dataframe: DataFrame) -> DataFrame:
    #     try:
    #         dropped_rows = "dropped_row"
    #         total_rows: int = dataframe.count()
    #         logger.info(f"Number of row: {total_rows} ")
    #
    #         # Drop row if target value is unknown
    #         logger.info(f"Dropping rows without target value.")
    #         unlabelled_dataframe: DataFrame = dataframe.filter(f"{self.target_column}== 'N/A'")
    #
    #         rejected_dir = os.path.join(self.data_validation_config.rejected_data_dir, dropped_rows)
    #         os.makedirs(rejected_dir, exist_ok=True)
    #         file_path = os.path.join(rejected_dir, self.data_validation_config.file_name)
    #
    #         unlabelled_dataframe = unlabelled_dataframe.withColumn(ERROR_MESSAGE, lit("Dropped row as target label is "
    #                                                                                   "unknown"))
    #
    #         logger.info(f"Unlabelled data has row: [{unlabelled_dataframe.count()}] and columns:"
    #                     f" [{len(unlabelled_dataframe.columns)}]")
    #
    #         logger.info(f"Write unlabelled data into rejected file path: [{file_path}]")
    #         unlabelled_dataframe.write.mode("append").parquet(file_path)
    #
    #         dataframe: DataFrame = dataframe.filter(f"{self.target_column}!= 'N/A'")
    #
    #         logger.info(f"Remaining data has rows: [{dataframe.count()}] and columns: [{len(dataframe.columns)}]")
    #         return dataframe
    #     except Exception as e:
    #         raise ConsumerComplaintException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Name: initiate_data_validation
        Description: Initiates data validation and preprocessing.

        This method reads the data, drops unwanted columns, checks for the presence of required columns,
        and saves the preprocessed data into the accepted data directory. It returns a DataValidationArtifact
        containing the path to the accepted data file and the rejected data directory.

        :return: A DataValidationArtifact object with file paths.
        :raises: ConsumerComplaintException if an error occurs during data validation or preprocessing.
        :version: 1.0
        """
        try:
            logger.info("Initiating data preprocessing.")
            dataframe: DataFrame = self.read_data()

            logger.info(f"Total Number of columns: {len(dataframe.columns)}")
            # dataframe = self.drop_row_without_target_label(dataframe=dataframe)

            logger.info("Dropping unwanted columns")
            dataframe: DataFrame = self.drop_unwanted_columns(dataframe=dataframe)

            column_count = len(dataframe.columns)

            # Print the column count
            logger.info(f"Number of columns in the DataFrame after deleting columns: {column_count}")

            # Validation to ensure that all required columns are available
            self.is_required_columns_exist(dataframe=dataframe)

            logger.info("Saving preprocessed data.")
            # Create the accepted data directory if it doesn't exist
            accepted_dir = Path(self.data_validation_config.accepted_data_dir)
            accepted_dir.mkdir(parents=True, exist_ok=True)
            accepted_file_path = accepted_dir / self.data_validation_config.file_name

            dataframe.write.mode("append").parquet(str(accepted_file_path))

            artifact = DataValidationArtifact(accepted_file_path=accepted_file_path,
                                            rejected_dir=self.data_validation_config.rejected_data_dir
                                            )
            logger.info(f"Data validation artifact: {artifact}")
            return artifact
        except Exception as e:
            raise ConsumerComplaintException(e, sys)

