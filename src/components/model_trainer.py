import os
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from ..exception import CustomException
from ..logger import logging

from ..utils import save_obj
from ..utils import evaluate_model   

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path =  os.path.join('artifacts', 'model')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info('Splitting ')

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostRegressor()' : GradientBoostingRegressor(),
            "AdaBoost" : AdaBoostRegressor(),
            'DecisionTreeRegressor' : DecisionTreeRegressor(),
            "SupportVectorRegressor" : SVR(),
            "KNN" : KNeighborsRegressor()
            }

            model_report:dict = evaluate_model(X_train,y_train, X_test, y_test, models)
            print(model_report)
            print("\n====================================================================================")
            logging.info(f'Model Report : {model_report}')

            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name :{best_model_name}, R2-score: {best_model_score}")
            print("\n====================================================================================")
            logging.info(f"Best Model Found, Model name: {best_model_name}, R2-score: {best_model_score}")
            logging.info(f"{best_model.feature_names_in_}")
            
            save_obj(
            file_path = self.model_trainer_config.trained_model_file_path,
            obj = best_model
            )

        except Exception as e:
            logging.info('Exception occured at model trianing')
            raise CustomException(e,sys)
        