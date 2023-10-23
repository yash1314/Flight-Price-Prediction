import sys
import os
from ..logger import logging
from ..exception import CustomException
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionconfig:
    data_path: str = os.path.join('notebooks/data', 'flight_fare.csv')
    data:str = os.path.join('artifact','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv(self.ingestion_config.data_path)
            logging.info('Dataset read as a pandas DataFrame')

            df.to_csv(self.ingestion_config.data,index=False,header=True)
            logging.info("Dataset stored in artifact folder as raw_data.csv")
            logging.info('Data ingestion is successful')
            return self.ingestion_config.data

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)
