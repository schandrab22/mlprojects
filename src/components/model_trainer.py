import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def intiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data.")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = {"Random Forest": RandomForestRegressor(),
                     "Decision Tree": DecisionTreeRegressor(),
                     "Linear Regression": LinearRegression(),
                     "K-Neighbor Regressor": KNeighborsRegressor(),
                     "XGBRegressor": XGBRegressor(),
                     "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                     "AdaBoost Regressor": AdaBoostRegressor(),
                     }
            params = {
                'Decision Tree':{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                'Random Forest':{
                    'n_estimators':[8,16,32,64,128,256],

                },
                'Gradient Boosting': {
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Linear Regression':{},
                'K-Neighbor Regressor':{
                    'n_neighbors':[5,6,7,8,9]
                },
                'XGBRegressor':{
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'CatBoosting Regressor': {
                    'depth': [6,8,10],
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'iterations': [30,50,100]
                },
                'AdaBoost Regressor': {
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                }

            }
            model_report:dict=evaluate_model(X_train=x_train, y_train=y_train,
                                             x_test = x_test, y_test = y_test,
                                               models=model, params=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = model[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            
            logging.info(f"Best found model on both training and testing dataset.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj = best_model
            )
            
            predicted = best_model.predict(x_test)
            return r2_score(y_test, predicted)
            # return r2_score


        except Exception as e:
            raise CustomException(e, sys)
