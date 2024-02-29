import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_train_data_path: str=os.path.join('data', "raw_train.csv")
    test_data_path: str=os.path.join('data', "test.csv")
    train_data_path: str=os.path.join('data', "train.csv")
    val_data_path: str=os.path.join('data', "val.csv")
    # raw_data_path: str=os.path.join('data', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try: 
            df_raw_train = pd.read_csv('/Users/gracebarringer/Machine Learning Projects/Kaggle/Obesity Risk - Multi-Class/Data/train.csv')
            logging.info('Read the raw training dataset as dataframe')
            df_test = pd.read_csv('/Users/gracebarringer/Machine Learning Projects/Kaggle/Obesity Risk - Multi-Class/Data/test.csv')
            logging.info('Read the testing dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_train_data_path), exist_ok = True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok = True)

            df_raw_train.to_csv(self.ingestion_config.raw_train_data_path, index = False, header = True)
            df_test.to_csv(self.ingestion_config.test_data_path)


            logging.info("Train test/val split initiated")
            df_train, df_val = train_test_split(df_raw_train, size = 0.2, random_state = 42)

            df_train.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            df_val.to_csv(self.ingestion_config.val_data_path, index = False, header = True)

            logging.info("Ingestion of data complete")

            return self.ingestion_config.train_data_path, self.ingestion_config.val_data_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
