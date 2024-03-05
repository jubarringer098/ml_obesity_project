import os
import sys
sys.path.insert(0, '/Users/gracebarringer/ml_project_code/ml_obesity_project')

import pandas as pd
from sklearn.pipeline import Pipeline 
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataTransformation
from src.exception import CustomException
from src.utils import load_object



try: 
    obj = DataIngestion()
    _, _, test_data = obj.initiate_data_ingestion()

    # Reading train (just for column sorting) and test data
    df_test = pd.read_csv(test_data)

    # Creating a pipeline object for data transformation
    pipeline_test = Pipeline([
        ('transform', DataTransformation(train=False))
    ])

    # Applying transformations on train and validation data
    df_test_trans = pipeline_test.fit_transform(df_test)

    # Proceeding to model training phase
    model_path=os.path.join("data","model.pkl")
    model = load_object(file_path = model_path)
    preds = model.predict(df_test_trans)

    # # Building final prediction file 
    id_list = df_test['id'].tolist()
    num_list = []
    for i in range(len(preds)):
        num_list.append(preds[i])

    df_preds = pd.DataFrame({"id":id_list, "NObeyesdad": num_list})
    df_preds.to_csv("/Users/gracebarringer/Machine Learning Projects/Kaggle/Obesity Risk - Multi-Class/preds.csv")


except Exception as e:
    raise CustomException(e, sys)
