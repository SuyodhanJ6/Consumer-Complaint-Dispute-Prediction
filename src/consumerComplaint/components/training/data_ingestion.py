import os
import re
import sys
import time
import uuid
import json
import pandas as pd
import requests
from collections import namedtuple
from typing import List
from datetime import datetime

from consumerComplaint.config.pipeline.training import FinanceConfig
from consumerComplaint.config.spark_manager import spark_session
from consumerComplaint.entity.metadata_entity import DataIngestionMetadata
from consumerComplaint.entity.artifact_entity import DataIngestionArtifact
from consumerComplaint.entity.config_entity import DataIngestionConfig
from consumerComplaint.logger import logger
from consumerComplaint.exception import ConsumerComplaintException


DownloadUrl = namedtuple("DownloadUrl", ["url", "file_path", "n_retry"])


class DataIngestion:
    """
    Name: DataIngestion
    Description: This class is used for data ingestion.
    Version: 1.0
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig, n_retry: int = 5):
        """
        Method Name: __init__
        Description: Initializes the DataIngestion object.
        
        :param data_ingestion_config: Data Ingestion configuration.
        :param n_retry: Number of retries in case of download failure.
        Version: 1.0
        """
        try:
            # Log an information message when initialization starts.
            logger.info("Initializing DataIngestion object...")
            
            self.data_ingestion_config = data_ingestion_config
            self.failed_download_urls: List[DownloadUrl] = []
            self.n_retry = n_retry

            # Log a success message when initialization completes.
            logger.info("DataIngestion object initialized successfully.")
            
        except Exception as e:
            # Log an error message if an exception occurs during initialization.
            # logger.exception("Error during DataIngestion initialization.")
            raise ConsumerComplaintException(e, sys)

    
    def get_required_interval(self):
        """
        Method Name: get_required_interval
        Description: Calculates the required time intervals based on data ingestion configuration.
        
        Output: List of time intervals as strings.
        
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            # Log an information message when interval calculation starts.
            logger.info("Calculating required time intervals...")
            
            start_date = datetime.strptime(self.data_ingestion_config.from_date, "%Y-%m-%d")
            end_date = datetime.strptime(self.data_ingestion_config.to_date, "%Y-%m-%d")
            n_diff_days = (end_date - start_date).days
            freq = None
            if n_diff_days > 365:
                freq = "Y"
            elif n_diff_days > 30:
                freq = "M"
            elif n_diff_days > 7:
                freq = "W"
            logger.info(f"{n_diff_days} hence freq: {freq}")
            if freq is None:
                intervals = pd.date_range(start=self.data_ingestion_config.from_date,
                                        end=self.data_ingestion_config.to_date,
                                        periods=2).astype('str').tolist()
            else:
                intervals = pd.date_range(start=self.data_ingestion_config.from_date,
                                        end=self.data_ingestion_config.to_date,
                                        freq=freq).astype('str').tolist()
            logger.info(f"Prepared Interval: {intervals}")
            if self.data_ingestion_config.to_date not in intervals:
                intervals.append(self.data_ingestion_config.to_date)
            
            # Log a success message when interval calculation completes.
            logger.info("Required time intervals calculated successfully.")
            
            return intervals
            
        except Exception as e:
        # Log an exception message if an exception occurs during interval calculation.
            raise ConsumerComplaintException(e, sys)

    
    def download_files(self, n_day_interval_url: int = None):
        """
        Method Name: download_files
        Description: Downloads data files for specified time intervals.
        
        :param n_day_interval_url: Number of day intervals for URL generation (optional).
        
        Output: List of DownloadUrl = namedtuple("DownloadUrl", ["url", "file_path", "n_retry"]).
        
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            # Calculate the required time intervals
            required_interval = self.get_required_interval()
            logger.info("Started downloading files")
            
            for index in range(1, len(required_interval)):
                from_date, to_date = required_interval[index - 1], required_interval[index]
                logger.debug(f"Generating data download url between {from_date} and {to_date}")
                
                # Generate the data source URL
                datasource_url: str = self.data_ingestion_config.datasource_url
                url = datasource_url.replace("<todate>", to_date).replace("<fromdate>", from_date)
                logger.debug(f"Url: {url}")
                
                # Define the file name and path
                file_name = f"{self.data_ingestion_config.file_name}_{from_date}_{to_date}.json"
                file_path = os.path.join(self.data_ingestion_config.download_dir, file_name)    
                
                # Create a DownloadUrl object
                download_url = DownloadUrl(url=url, file_path=file_path, n_retry=self.n_retry)
                
                # Download the data
                self.download_data(download_url=download_url)
            
            logger.info(f"File download completed")
            
        except Exception as e:
            # Log an exception message if an exception occurs during file download.
            # logger.exception("Error during file download.")  # Removed as requested
            raise ConsumerComplaintException(e, sys)

    
    def convert_files_to_parquet(self) -> str:
        """
        Method Name: convert_files_to_parquet
        Description: Downloads files will be converted and merged into a single Parquet file.
        
        Output: Path to the output Parquet file.
        
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            # Define the directories and output file name
            json_data_dir = self.data_ingestion_config.download_dir
            data_dir = self.data_ingestion_config.feature_store_dir
            output_file_name = self.data_ingestion_config.file_name
            
            # Create the output directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Define the output file path
            file_path = os.path.join(data_dir, f"{output_file_name}")
            logger.info(f"Parquet file will be created at: {file_path}")
            
            # Check if the JSON data directory exists
            if not os.path.exists(json_data_dir):
                return file_path
            
            # Iterate through JSON files and convert them to Parquet
            for file_name in os.listdir(json_data_dir):
                json_file_path = os.path.join(json_data_dir, file_name)
                logger.debug(f"Converting {json_file_path} into Parquet format at {file_path}")
                
                # Read JSON data and convert to DataFrame
                df = spark_session.read.json(json_file_path)
                
                # Append the DataFrame to the Parquet file if it contains data
                if df.count() > 0:
                    df.write.mode('append').parquet(file_path)
            
            return file_path
    
        except Exception as e:
            # Log an exception message if an exception occurs during the conversion.
            # logger.exception("Error during file conversion.")  # Removed as requested
            raise ConsumerComplaintException(e, sys)


    def retry_download_data(self, data, download_url: DownloadUrl):
        """
        Method Name: retry_download_data
        Description: Helps to avoid download failures by retrying the download if possible.
        
        Parameters:
        - data: Failed response from the initial download attempt.
        - download_url: DownloadUrl object containing download details.
        
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            # Log that the retry_download_data method is starting.
            logger.info("Starting retry_download_data method...")

            # Check if retry is still possible, otherwise, add the URL to failed downloads
            if download_url.n_retry == 0:
                self.failed_download_urls.append(download_url)
                logger.info(f"Unable to download file {download_url.url}")
                return

            # Handle throttling requests by waiting for some seconds if indicated in the response
            content = data.content.decode("utf-8")
            wait_seconds = re.findall(r'\d+', content)

            if len(wait_seconds) > 0:
                time.sleep(int(wait_seconds[0]) + 2)

            # Write the response to a failed file for analysis
            failed_file_path = os.path.join(self.data_ingestion_config.failed_dir,
                                            os.path.basename(download_url.file_path))
            os.makedirs(self.data_ingestion_config.failed_dir, exist_ok=True)
            with open(failed_file_path, "wb") as file_obj:
                file_obj.write(data.content)

            # Retry the download by decrementing the retry count
            download_url = DownloadUrl(download_url.url, file_path=download_url.file_path,
                                    n_retry=download_url.n_retry - 1)
            self.download_data(download_url=download_url)
            
            # Log that the retry_download_data method has completed.
            logger.info("retry_download_data method completed successfully.")
        
        except Exception as e:
            # Log an exception message if an exception occurs during retry.
            # logger.exception("Error during download retry.")
            raise ConsumerComplaintException(e, sys)


    def download_data(self, download_url: DownloadUrl):
        """
        Method Name: download_data
        Description: Downloads data from the provided URL and writes it to a JSON file.
        
        Parameters:
        - download_url: DownloadUrl object containing download details.
        
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            # Log that the download_data method is starting.
            logger.info(f"Starting download operation: {download_url}")
            
            download_dir = os.path.dirname(download_url.file_path)

            # Creating download directory if it doesn't exist
            os.makedirs(download_dir, exist_ok=True)

            # Downloading data
            data = requests.get(download_url.url, params={'User-agent': f'your bot {uuid.uuid4()}'})

            try:
                # Log that writing downloaded data to a JSON file has started.
                logger.info(f"Started writing downloaded data into JSON file: {download_url.file_path}")
                
                # Saving downloaded data into a JSON file on disk
                with open(download_url.file_path, "w") as file_obj:
                    finance_complaint_data = list(map(lambda x: x["_source"],
                                                    filter(lambda x: "_source" in x.keys(),
                                                            json.loads(data.content)))
                                                )

                    json.dump(finance_complaint_data, file_obj)
                    
                # Log that downloaded data has been successfully written to the file.
                logger.info(f"Downloaded data has been written into file: {download_url.file_path}")
            
            except Exception as e:
                # Log a message indicating that download has failed and a retry will be attempted.
                logger.info("Failed to download data, hence retrying...")
                
                # Remove the failed file if it exists
                if os.path.exists(download_url.file_path):
                    os.remove(download_url.file_path)
                
                # Retry the download
                self.retry_download_data(data, download_url=download_url)

        except Exception as e:
            # Log an exception message if an exception occurs during download.
            # logger.exception("Error during download operation.")
            raise ConsumerComplaintException(e, sys)

    def write_metadata(self, file_path: str) -> None:
        """
        Method Name: write_metadata
        Description: Updates metadata information to avoid redundant download and merging.

        Parameters:
        - file_path: Path to the downloaded and merged data file.

        On Failure: Writes an exception log and then raises an exception.

        Version: 1.0
        """
        try:
            # Log that metadata info is being written to the metadata file.
            logger.info(f"Writing metadata info into metadata file.")
            
            # Create a DataIngestionMetadata object
            metadata_info = DataIngestionMetadata(metadata_file_path=self.data_ingestion_config.metadata_file_path)

            # Write metadata information to the metadata file
            metadata_info.write_metadata_info(from_date=self.data_ingestion_config.from_date,
                                            to_date=self.data_ingestion_config.to_date,
                                            data_file_path=file_path
                                            )
            
            # Log that metadata has been successfully written.
            logger.info(f"Metadata has been written.")
        
        except Exception as e:
            # Log an exception message if an exception occurs during metadata writing.
            # logger.exception("Error during metadata writing.")
            raise ConsumerComplaintException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: Initiates the data ingestion process, including downloading, conversion, and metadata update.

        Returns: DataIngestionArtifact object containing information about the downloaded data.

        On Failure: Writes an exception log and then raises an exception.

        Version: 1.0
        """
        try:
            # Log that the downloading of JSON files has started.
            logger.info(f"Started downloading JSON files")
            
            # Check if from_date and to_date are not the same (to avoid redundant download)
            if self.data_ingestion_config.from_date != self.data_ingestion_config.to_date:
                self.download_files()

            # Check if the download directory exists
            if os.path.exists(self.data_ingestion_config.download_dir):
                # Log that the conversion and combination of downloaded JSON into a Parquet file has started.
                logger.info(f"Converting and combining downloaded JSON into a Parquet file")
                
                # Convert and combine downloaded JSON into a Parquet file
                file_path = self.convert_files_to_parquet()
                
                # Write metadata information for the merged data file
                self.write_metadata(file_path=file_path)

            # Define the path to the feature store file
            feature_store_file_path = os.path.join(self.data_ingestion_config.feature_store_dir,
                                                self.data_ingestion_config.file_name)

            # Create a DataIngestionArtifact object containing relevant information
            artifact = DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path,
                download_dir=self.data_ingestion_config.download_dir,
                metadata_file_path=self.data_ingestion_config.metadata_file_path,
            )

            # Log the data ingestion artifact
            logger.info(f"Data ingestion artifact: {artifact}")
            logger.info(f"<-------------------------------- Data ingestion is completed successfully -------------------------------->")
            return artifact
        

            logger.info(f"Data ingestion is completed successfully")
        
        except Exception as e:
            # Log an exception message if an exception occurs during data ingestion.
            # logger.exception("Error during data ingestion.")
            raise ConsumerComplaintException(e, sys)

        

# def main():
#     try:
#         config = FinanceConfig()
#         data_ingestion_config = config.get_data_ingestion_config()
#         data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
#         data_ingestion.initiate_data_ingestion()
#     except Exception as e:
#         raise ConsumerComplaintException(e, sys)


# if __name__ == "__main__":
#     try:
#         main()

#     except Exception as e:
#         logger.exception(e)

