# Bank Customers Churn

## Objectives
1. Every bank wants to keep its customers in order to maintain its business, so Multinational Bank ABC decided to work on Multinational Bank ABC's customer data. We are working on Multinational Bank ABC's account holder data and the purpose of this data is to predict the customer Churn.
2. To solve the problem statement, I implemented several experiments using classification algorithms (KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, XGBoostClassifier).
3. To monitor each experiment, I used mlflow.

### Tools & technologies used
## 1. Programming language : Python
<div align="center">
  <img src="https://github.com/Ahak99/used-car-price/assets/101395769/77eb34b4-d758-4f70-bbf9-4cde54ced129" alt="Alt Text">
</div>

## 2. ML libraries : Tensorflow, Keras, Sklearn, pandas
<div align="center">
  <img src="https://github.com/Ahak99/used-car-price/assets/101395769/fae06a0b-7055-4c42-85f0-3a424bad9bef" alt="Alt Text">
</div>

## 3. MLflow
<div align="center">
  <img src="https://github.com/Ahak99/Bank-Customers-Churn/assets/101395769/123c4899-a6a7-4324-8176-560139b8ff29" alt="Alt Text">
</div>

## 4. Github
<div align="center">
  <img src="https://github.com/Ahak99/used-car-price/assets/101395769/308b6f2c-6e69-4c92-b210-9d82b2d257e3" alt="Alt Text">
</div>


## Life cycle of project
    1. Install the dependencies
    2. Collect data
        - Dataset Source - https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data
    3. Read data
    4. Data Checks to perform
        - Check Missing values.
        - Check Duplicates.
        - Check data type.
        - Check the number of unique values of each column.
        - Check statistics of data set.
        - Check various categories present in the different categorical column.

    3. Data Analysis & Visualisation (more details in the notebook)

    5. Model building
        - Split data into Training/Testing sets.
        - Normalise the data.
        - Run several experiments using classification algorithms (KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, XGBoostClassifier).
        - Use mlflow to control each experiment.
        - Select the best model for each experiment
        - Save the models
        - Use the best model from all experiments
        - Predict

    6. Push the project to GitHub

## Project folder
	|---- artifacts
	|---- BEST_MODELS
	|---- data
	    |--- Bank Customer Churn Prediction.csv
	|---- logs
	|---- mlruns
	|---- Notebook
	|---- src
	    |--- experiences
	         |--- __init__.py
		 |--- AdaBoostClassifier_experience.py
	  	 |--- DecisionTree_experience.py
	         |--- KNeighborsClassifier_experience.py
	         |--- RandomForestClassifier_experience.py
	         |--- svc_experience.py
	         |--- XGBClassifier_experience.py
	    |--- pipeline
	         |--- __init__.py
	         |--- data_prep.py
	         |--- model_selector.py
	         |--- Training.py
	    |--- __init__.py
	    |--- exception.py
	    |--- logger.py
	    |--- utils.py
	|---- venv
	|---- experiments_infos.csv
	|---- README.md
	|---- requirements.txt


## Software and tools requirements

    1. Github
    2. Any IDE to work (VScode, PyCharm ...)
    3. Heroku account

## Create a new environment

    python -m venv .venv

## Activate the environment

    .venv/Scripts/activate
