import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import  CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    pre_processor_obj_file_path = os.path.join('artifact','preprocesor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_object(self):
        '''
            This function is responsible for data transformation.
        '''
        try:
            neumarical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education", 
                "lunch", "test_preparation_course"
            ]
            numerical_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ("scaler", SimpleImputer())
                    ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorial columns : {categorical_columns}")
            logging.info(f"Neumarical columns : {neumarical_columns}")

            pre_processor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipline, neumarical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return pre_processor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transform(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed.")

            preprocessor_obj = self.get_data_transform_object()

            target_column = 'math_score'
            neumarical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved processing object.")

            save_object(
                file_path = self.data_transformation_config.pre_processor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr, test_arr, self.data_transformation_config.pre_processor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)