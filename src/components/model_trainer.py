import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object,evaluate_model
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_filepath=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer=ModelTrainerConfig()

    def initiate_model_train(self,train_array,test_array):
        try:
            logging.info("model training starts")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor(),
                
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'SVR': SVR(),
                'Ridge': Ridge(),
                'RandomForest':RandomForestRegressor()

            }

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer.trained_model_filepath,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            #r2_square=r2_score(y_test,predicted)
            #return r2_square

            return(predicted,y_test)

        except Exception as e:
            raise CustomException(e,sys)
