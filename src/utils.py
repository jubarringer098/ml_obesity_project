import os
import sys

from src.exception import CustomException

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import pickle


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_val, y_val, models, param):
    try:
        report = {}
        y_train = y_train.squeeze()
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]
            grid_search = GridSearchCV(model, params, cv = 3)
            grid_search.fit(X_train, y_train)
            # model.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)


            y_train_pred = model.predict(X_train)

            y_val_pred = model.predict(X_val)

            train_model_accuracy = accuracy_score(y_train, y_train_pred)
            # train_model_precision = metrics.precision_score(y_train, y_train_pred,  average = 'micro')
            # train_model_recall = metrics.recall_score(y_train, y_train_pred,  average = 'micro')
            # train_model_f1 = metrics.f1_score(y_train, y_train_pred, average = 'micro')
            # train_model_auc = metrics.roc_auc_score(y_train, y_train_pred)

            val_model_accuracy = accuracy_score(y_val, y_val_pred)
            # val_model_precision = metrics.precision_score(y_val, y_val_pred,  average = 'micro')
            # val_model_recall = metrics.recall_score(y_val, y_val_pred,  average = 'micro')
            # val_model_f1 = metrics.f1_score(y_val, y_val_pred, average = 'micro')
            # val_model_auc = metrics.roc_auc_score(y_val, y_val_pred)

            report[list(models.keys())[i]] = val_model_accuracy

            return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)