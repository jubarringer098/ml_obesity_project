import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
# from sklearn.multiclass
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, r2_score, classification_report, roc_auc_score, roc_curve


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('data', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, df_train, df_val):
        try:
            logging.info("Splitting training and validation input data")

            X_train, y_train, X_val, y_val = (
                df_train.iloc[:,:-1],
                df_train.iloc[:,-1:],
                df_val.iloc[:,:-1],
                df_val.iloc[:,-1:],
            )

            models = {
                "Random Forest" : RandomForestClassifier(random_state = 55),
                "XGBClassifier" : XGBClassifier(random_state = 55),
                "CatBoostClassifier" : CatBoostClassifier(random_state = 55),
                "AdaBoostClassifier" : AdaBoostClassifier(random_state = 55)
            }

            
            params = {
                "Random Forest" : {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [7, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "XGBClassifier" : {
                    'learning_rate':[.1,.01,.05],
                    'max_depth': [5, 7, 10],
                    'subsample':[0.7,0.8,0.9],
                    'n_estimators': [100, 300, 500]
                },
                "CatBoostClassifier" : {
                    'depth': [4,6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 150]
                },
                "AdaBoostClassifier" : {
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [25, 50, 100, 200]
                }
            }


            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, models = models, param = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")

            logging.info("Best model on both training and validation datasets")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            logging.info("Best model saved to pkl file")


            train_predicted = best_model.predict(X_train)

            train_accuracy_score = accuracy_score(y_train, train_predicted)

            val_predicted = best_model.predict(X_val)

            val_accuracy_score = accuracy_score(y_val, val_predicted)

            return train_accuracy_score, val_accuracy_score, best_model


        except Exception as e:
            raise CustomException(e, sys)