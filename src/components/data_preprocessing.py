import sys
import os
# from dataclass import dataclass 
sys.path.insert(0, '/Users/gracebarringer/ml_project_code/ml_obesity_project')

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# @dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            
            cont_vars = ['Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Age_bc']
            cat_vars = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC','SMOKE', 'SCC', 'CALC','MTRANS']
            target_var = "NObeyesdad"
            # cont_pipeline = Pipeline(
            #     steps = [

            #     ]
            # )

            cat_pipeline = Pipeline(
                steps = [
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            target_pipeline = Pipeline(
                steps = [
                    ('label_encoder', LabelEncoder())  
                ]
            )

            logging.info("Categorical and target encoding complete")

            preprocessor = ColumnTransformer(
                [
                    ("cat_pipeline", cat_pipeline, cat_vars),
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, train_path, val_path):

        try:
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)

            logging.info("Read train and validation data complete")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_var = "NObeyesdad"
  

            input_feature_df_train = df_train.drop(columns = [target_var, 'id'], axis = 1)
            # target_feature_df_train = df_train[target_var]
            target_feature_df_train_encoded = LabelEncoder().fit_transform(df_train[target_var])

            input_feature_df_val = df_val.drop(columns = [target_var, 'id'], axis = 1)
            # target_feature_df_val = df_val[target_var]
            target_feature_df_val_encoded = LabelEncoder().fit_transform(df_val[target_var])


            logging.info("Applying preprocessing object on training and validation dataframe")


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_df_train)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_df_val)




            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_df_train_encoded)
            ]
            val_arr = np.c_[
                input_feature_val_arr, np.array(target_feature_df_val_encoded)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return train_arr, val_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)