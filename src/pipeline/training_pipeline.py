import os
import sys
from src.components.data_ingestion import DataIngestionConfig,dataingestion
from src.components.data_transformation import DataTransformationConfig,DataTransformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.components.model_evaluation import ModelEvaluationConfig
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException



try:

    if __name__=="__main__":
        obj= dataingestion()
        train_data,test_data=obj.initiate_data_ingestion()
        
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

        model_trainer=ModelTrainer()
        y_test,predicted=model_trainer.initiate_model_train(train_arr,test_arr)
        print('r2_score',"this is score")

        model_evaluation=ModelEvaluationConfig()
        rmse,mae,r2_score=model_evaluation.eval_metrics(predicted,y_test)
        print(rmse,mae,r2_score,"in modelevaluation")

except Exception as e:
    raise CustomException(e,sys)