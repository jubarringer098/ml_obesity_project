import sys
import os
from dataclasses import dataclass 
sys.path.insert(0, '/Users/gracebarringer/ml_project_code/ml_obesity_project')

import numpy as np
import pandas as pd
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox

from src.exception import CustomException
from src.logger import logging

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('data', 'preprocessor.pkl')

class DataTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, train = True):
        self.train = train
        self.cat_vars = None
        self.cont_vars = None
        self.le = LabelEncoder() if train else None


    def fit(self, X, y = None):
        try:
            if self.train == True: 
                self.cat_vars = X.select_dtypes(include = 'object').columns.tolist()
                self.cont_vars = X.select_dtypes(exclude = 'object').columns.tolist()
                self.cont_vars.remove('id')
                self.cat_vars.remove('NObeyesdad')
                
                self.le.fit(X['NObeyesdad'])
            else: 
                self.cat_vars = X.select_dtypes(include = 'object').columns.tolist()
                self.cont_vars = X.select_dtypes(exclude = 'object').columns.tolist()
                self.cont_vars.remove('id')
        
            logging.info(f"Categorical variables: {self.cat_vars}")
            logging.info(f"Continuous variables: {self.cont_vars}")
   
            return self
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform(self, X):

        try:
            X_transformed = X.copy()
            
            # Apply Box-Cox transformation
            X_transformed['Age_bc'], fitted_lambda = boxcox(X_transformed['Age']+1)
            X_transformed = X_transformed.drop(columns = ['Age', 'id'], axis = 1)

            # Calculate BMI
            X_transformed['BMI'] = X_transformed['Weight']/X_transformed['Height']**2


            logging.info("Continous transformations complete")

            # Encode categorical variables
            X_transformed = pd.get_dummies(X_transformed, columns=self.cat_vars, dtype=int)

            logging.info("Categorical encoding complete")

            # Apply label encoding to 'NObeyesdad' and adding missing category classes if in train mode
            if self.train == True:
                X_transformed['NObeyesdad'] = self.le.transform(X_transformed['NObeyesdad'])
                logging.info("Target encoding complete")

                X_transformed['CALC_Always'] = 0
                X_transformed['MTRANS_Walking'] = 0

                # Sorting dataset
                X_transformed = X_transformed.sort_index(axis =1)
                X_transformed['NObeyesdad'] = X_transformed.pop('NObeyesdad')
            else:
                # Sorting dataset
                X_transformed = X_transformed.sort_index(axis =1)
        
            logging.info("Data transformations complete")

            return X_transformed
        except Exception as e:
            raise CustomException(e, sys)

# pipeline = Pipeline([('transform', DataTransformation(train=True))])



