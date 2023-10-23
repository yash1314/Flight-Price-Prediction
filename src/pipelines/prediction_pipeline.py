import sys
import os
from ..exception import CustomException
from ..utils import load_object
from ..logger import logging

import pandas as pd
import numpy as np

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            ## model loading 
            model_path = os.path.join('artifacts', 'model')
            
            ## model object creation 
            model = load_object(model_path)
            
            ## prediction
            pred = model.predict(features)
            return pred
        
        except Exception as e:
            logging.info('Error occured in predict pipeline folder')
            raise CustomException(e, sys)
        