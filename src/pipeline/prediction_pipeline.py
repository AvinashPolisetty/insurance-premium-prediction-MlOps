import os
import sys
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor= load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 age: float,
                 sex:object,
                 bmi:float,
                 children:float,
                 smoker:object,
                 region:object):
        
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region=region


        def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'age':[self.age],
                    'sex':[self.sex],
                    'bmi':[self.bmi],
                    'children':[self.children],
                    'smoker':[self.smoker],
                    'region':[self.region],
                
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise CustomException(e,sys)
