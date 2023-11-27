import sys
sys.path.append('D:\WORK\Personnel\Python projects\GitHub projects\Bank-Customers-Churn')
import pandas as pd

from src.experiences import AdaBoostClassifier_experience, DecisionTree_experience, KNeighborsClassifier_experience, RandomForestClassifier_experience, svc_experience, XGBClassifier_experience

from src.pipeline.data_prep import DataPrep
from src.pipeline.model_selector import best_model


if __name__ == "__main__":
    
    data_path = "data\Bank Customer Churn Prediction.csv"
    Data_Prep = DataPrep()
    X_train, X_test, y_train, y_test, _ = Data_Prep.initiate_data_transfromation(data_path)   
    
    # print("\n######################################    Experience 1   ######################################\n")
    # KNeighborsClassifier_experience.KNeighborsClassifier_experience(X_train, X_test, y_train, y_test)
    # print("\n######################################    Experience 2   ######################################\n")
    # DecisionTree_experience.Decision_Tree_experience(X_train, X_test, y_train, y_test)
    # print("\n######################################    Experience 3   ######################################\n")
    # RandomForestClassifier_experience.RandomForestClassifier_experience(X_train, X_test, y_train, y_test)
    # print("######################################    Experience 4   ######################################\n")
    # svc_experience.SupportVectorClassifier_experience(X_train, X_test, y_train, y_test)
    # print("######################################    Experience 5   ######################################\n")
    # AdaBoostClassifier_experience.AdaBoostClassifier_experience(X_train, X_test, y_train, y_test)
    # print("\n######################################    Experience 6   ######################################\n")
    # XGBClassifier_experience.XGBClassifier_experience(X_train, X_test, y_train, y_test)
    
    EXPERIMENTS = [ "KNeighborsClassifier - Experiment", "DecisionTreeClassifier - Experiment", "RandomForestClassifier - Experiment", "SupportVectorClassifier - Experiment", "AdaBoostClassifier - Experiment","XGBoostClassifier - Experiment"]

    best_predictor = best_model(EXPERIMENTS)