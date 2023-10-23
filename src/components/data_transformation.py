import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt 


import numpy as np
import pandas as pd

from ..exception import CustomException
from ..logger import logging
from ..utils import * 

from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    transformed_data_file_path = os.path.join('artifact', 'transformed_data.csv')
    

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


   
    def initiate_data_transformation(self, data_path):
        try : 
            ## reading the data
            df = pd.read_csv(data_path)

            logging.info('Read data completed')
            logging.info(f'df dataframe head: \n{df.head().to_string()}')

            ## dropping null values
            df.dropna(inplace = True)

            ## Date of journey column transformation
            df['journey_date'] = pd.to_datetime(df['Date_of_Journey'], format ="%d/%m/%Y").dt.day
            df['journey_month'] = pd.to_datetime(df['Date_of_Journey'], format ="%d/%m/%Y").dt.month

            ## encoding total stops.
            df.replace({'Total_Stops': {'non-stop' : 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}}, inplace = True)

            ## ecoding airline, source, and destination
            df_airline = pd.get_dummies(df['Airline'], dtype=int)
            df_source = pd.get_dummies(df['Source'],  dtype=int)
            df_dest = pd.get_dummies(df['Destination'], dtype=int)

            ## dropping first columns of each categorical variables.
            df_airline.drop('Trujet', axis = 1, inplace = True)
            df_source.drop('Banglore', axis = 1, inplace = True)
            df_dest.drop('Banglore', axis = 1, inplace = True)

            df = pd.concat([df, df_airline, df_source, df_dest], axis = 1)

            ## handling duration column
            df['duration'] = df['Duration'].apply(convert_to_minutes)
            upper_time_limit = df.duration.mean() + 1.5 * df.duration.std()
            df['duration'] = df['duration'].clip(upper = upper_time_limit)

            ## encodign duration column
            bins = [0, 120, 360, 1440]  # custom bin intervals for 'Short,' 'Medium,' and 'Long'
            labels = ['Short', 'Medium', 'Long'] # creating labels for encoding

            df['duration'] = pd.cut(df['duration'], bins=bins, labels=labels)
            df.replace({'duration': {'Short':1, 'Medium':2, 'Long': 3}}, inplace = True)
            
            ## dropping the columns
            cols_to_drop = cols_to_drop = ['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Additional_Info', 'Delhi', 'Kolkata']

            df.drop(cols_to_drop, axis = 1, inplace = True)

            logging.info('df data transformation completed')
            logging.info(f' transformed df data head: \n{df.head().to_string()}')

            df.to_csv(self.data_transformation_config.transformed_data_file_path, index = False, header= True)
            logging.info("transformed data is stored")
            df.head(1)
            ## splitting the data into training and target data
            X = df.drop('Price', axis = 1)
            y = df['Price']
            
            ## accessing the feature importance.
            select = ExtraTreesRegressor()
            select.fit(X, y)

            plt.figure(figsize=(12, 8))
            fig_importances = pd.Series(select.feature_importances_, index=X.columns)
            fig_importances.nlargest(20).plot(kind='barh')
        
            ## specify the path to the "visuals" folder using os.path.join
            visuals_folder = 'visuals'
            if not os.path.exists(visuals_folder):
                os.makedirs(visuals_folder)

            ## save the plot in the visuals folder
            plt.savefig(os.path.join(visuals_folder, 'feature_importance_plot.png'))
            logging.info('feature imp figure saving is successful')

            ## further Splitting the data.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True) 
            logging.info('final splitting the data is successful')
            
            ## returning splitted data and data_path.
            return (
                X_train, 
                X_test, 
                y_train, 
                y_test,
                self.data_transformation_config.transformed_data_file_path
            )
        
        except Exception as e:
            logging.info('error occured in the initiate_data_transformation')
            raise CustomException(e, sys)
