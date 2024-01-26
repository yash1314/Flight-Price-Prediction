import os
import pandas as pd
import numpy as np

import sys
import joblib
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj, compress= ('gzip'))

    except Exception as e:
        logging.info('Error occured in utils save_obj')
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):

    try:
        report = {}
        for i in range(len(models)):

            model = list(models.values())[i]

            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_obj in utils')
        raise CustomException(e, sys)
             
def convert_to_minutes(duration):
    try:
        hours, minute = 0, 0
        for i in duration.split():
            if 'h' in i:
                hours = int(i[:-1])
            elif 'm' in i:
                minute = int(i[:-1])
        return hours * 60 + minute
    except :
        return None


def preprocess(Airline, Date_of_Journey, Source, Destination, Duration, Total_Stops):
    try:
        my_cols = {'Total_Stops' :0, 'journey_date' :0,'journey_month' :0,'Air Asia' :0,'Air India':0, 'GoAir':0, 'IndiGo':0, 'Jet Airways':0, 'Jet Airways Business':0, 'Multiple carriers':0,
    'Multiple carriers Premium economy':0, 'SpiceJet':0, 'Vistara':0,
    'Vistara Premium economy':0, 'Chennai':0, 'Mumbai':0, 'Cochin':0, 'Hyderabad':0, 'New Delhi':0, 'duration':0}
        
        new_cols = my_cols.copy()

        my_cols['journey_date'], my_cols['journey_month'] = pd.to_datetime(Date_of_Journey).day, pd.to_datetime(Date_of_Journey).month

        my_cols['duration'] = {'Short': 1, 'Medium': 2, 'Long': 3}.get(Duration, 0)
        my_cols['Total_Stops'] = {'non-stop' : 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}.get(Total_Stops, 0)
        
        if Destination == 'Banglore': 
            pass
        else:
            my_cols[f'{Destination}'] = 1
            
        if Source == 'Banglore':
            pass
        else:
            my_cols[f'{Source}'] = 1
            
        if Airline == 'Trujet':
            pass
        else:
            my_cols[f"{Airline}"] = 1
            
        
        df = pd.DataFrame(data = my_cols, index = [0])
        my_cols = new_cols

        return df
    
    except Exception as e:
        logging.info('Error occured in user input data preprocessing.')
        raise CustomException(e, sys)