import os
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
import joblib
import sys
from dataclasses import dataclass
from src.components.model_trainer import *
from src.logger import logging
import pandas as pd


@dataclass
class ModelEvaluationConfig:
    
    def eval_metrics(self,actual,pred):
        try:
            rmse=np.round(mean_squared_error(actual,pred))
            mae=np.round(mean_squared_error(actual,pred))
            r2=r2_score(actual,pred)

            return rmse,mae,r2
        except Exception as e:
            raise CustomException(e,sys)
    

    def log_into_mlflow(self):
        
        try:
            test_data=pd.read_csv(os.path.join('artifacts','test.csv'))
            model=joblib.load(os.path.join('artifacts','model.pkl'))

            test_x0 = test_data.drop('expenses',axis=1)
            test_y = test_data['expenses']

            test_x=pd.get_dummies(test_x0)

            mlflow.set_registry_uri("https://dagshub.com/avinash8/insurance-premium-prediction-MlOps.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                pred=model.predict(test_x)

                (rmse,mae,r2)=self.eval_metrics(test_y,pred)
                # Saving metrics as local
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                print(scores)

                params={'n_estimators':100,'min_samples_split':2, 'min_samples_leaf':1}

                mlflow.log_params(params)
                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("mae",mae)
                mlflow.log_metric("r2",r2)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestModel")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise CustomException(e,sys)