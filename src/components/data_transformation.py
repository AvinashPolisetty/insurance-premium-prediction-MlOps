import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import *


@dataclass
class DataTransformationConfig:
    preprocessor_filepath=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_tranformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("data transformation initiated")

            numerical_columns = ['age', 'bmi', 'children']
            categorical_columns = ['sex', 'smoker', 'region']

            sex_categories = ['male', 'female']
            smoker_categories = ['yes', 'no'] 
            region_categories = ['southwest', 'southeast', 'northwest', 'northeast']




            num_pipeline=Pipeline(
                steps=[
                    ("scalar",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    
                    ('onehotencoding',OneHotEncoder(categories=[sex_categories,
                                                                smoker_categories,region_categories])),
                    
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading the train and test data completed")
            logging.info("obtaining the preprocessing object")

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name = 'expenses'
            drop_columns = [target_column_name]

            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_tranformation_config.preprocessor_filepath,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_filepath,
            )

            
        except Exception as e:
            raise CustomException(e,sys)
    

