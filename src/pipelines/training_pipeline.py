import os
import sys
from ..logger import logging
from ..exception import CustomException
import pandas as pd
import numpy as np

from ..components.data_ingestion import DataIngestion
from ..components.data_transformation import DataTransformation
from ..components.model_trainer import ModelTrainer

if __name__=='__main__':
    # creating a data ingestion object
    obj = DataIngestion()

    # getting a data_path from data ingestion
    data_path = obj.initiate_data_ingestion()

    # creating a data transformation object
    data_transformation = DataTransformation()

    #getting splitted data.
    X_train, X_test, y_train, y_test, _ = data_transformation.initiate_data_transformation(data_path= data_path)

    # passing splitted data into model training pipeline
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(X_train= X_train, X_test= X_test, y_train= y_train, y_test= y_test)
