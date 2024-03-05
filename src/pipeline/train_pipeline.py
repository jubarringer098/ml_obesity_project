import sys
import os
sys.path.insert(0, '/Users/gracebarringer/ml_project_code/ml_obesity_project')

import pandas as pd
from sklearn.pipeline import Pipeline 
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

try: 
    obj = DataIngestion()
    train_data, val_data, _ = obj.initiate_data_ingestion()


    # Reading train and validation data
    df_train = pd.read_csv(train_data)
    df_val = pd.read_csv(val_data)

    # Creating a pipeline object for data transformation
    pipeline_train_val = Pipeline([
        ('transform', DataTransformation(train=True))
    ])

    # Applying transformations on train and validation data
    df_train = pipeline_train_val.fit_transform(df_train)
    df_val = pipeline_train_val.transform(df_val)  # Note: Use transform() here instead of fit_transform() to avoid data leakage from the validation set

    # Proceeding to model training phase
    model_trainer = ModelTrainer() 
    print(model_trainer.initiate_model_trainer(df_train, df_val))

except Exception as e:
    raise CustomException(e, sys)
