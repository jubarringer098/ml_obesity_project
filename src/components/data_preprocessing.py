import sys
import os
# from dataclass import dataclass 
sys.path.insert(0, '/Users/gracebarringer/ml_project_code/ml_obesity_project')

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# @dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data', 'preprocessor.pkl')


class DataTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, train = True):
        self.train = train
        self.cat_vars = None
        self.cont_vars = None
        self.le = LabelEncoder() if train else None
        self.data_transformation_config = DataTransformationConfig()


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
        
        
            return self
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform(self, X):

        try:
            X_transformed = X.copy()
            
            # Apply Box-Cox transformation
            X_transformed['Age_bc'], fitted_lambda = boxcox(X_transformed['Age']+1)
            X_transformed = X_transformed.drop(columns = ['Age'], axis = 1)
            # Calculate BMI
            X_transformed['BMI'] = X_transformed['Weight']/X_transformed['Height']**2

            # Encode categorical variables
            X_transformed = pd.get_dummies(X_transformed, columns=self.cat_vars, dtype=int)
            
            # Apply label encoding to 'NObeyesdad' if in train mode
            if self.train == True:
                X_transformed['NObeyesdad'] = self.le.transform(X_transformed['NObeyesdad'])
            

            return X_transformed
        except Exception as e:
            raise CustomException(e, sys)

# pipeline = Pipeline([('transform', DataTransformation(train=True))])



