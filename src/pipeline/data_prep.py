import os
import sys
sys.path.append('D:\WORK\Personnel\Python projects\GitHub projects\Bank-Customers-Churn')
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataPrepConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataPrep:
    def __init__(self):
        self.data_transformation_config = DataPrepConfig()

    def get_data_transfromer_object(self):
        
        try:
            numerical_features = ["credit_score", "age", "tenure", "balance", "products_number", "credit_card", "active_member", "estimated_salary"]
            categorical_features = ["country", "gender"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder())
                ]
            )

            logging.info(f"Numerical features : {numerical_features}")
            logging.info(f"Categorical features : {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_features),
                    ("cat_pipeline",cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)
        
    
    def initiate_data_transfromation(self, data_path):
        
        try:
            data = pd.read_csv(data_path)

            X = data.drop(["customer_id", "churn"], axis=1)
            y = data["churn"]
            
            input_feature_train_df, input_feature_test_df, target_train_df, target_test_df = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Read data and split it into train and test sets - completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transfromer_object()

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)


            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                input_feature_train_arr,
                input_feature_test_arr,
                target_train_df, 
                target_test_df,                
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)