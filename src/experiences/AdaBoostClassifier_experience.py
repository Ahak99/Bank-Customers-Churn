import sys
sys.path.append(r'D:\WORK\Personnel\Python projects\GitHub projects\Bank-Customers-Churn')
import time
import random
import concurrent.futures
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from src.utils import train


def AdaBoostClassifier_experience(X_train, X_test, y_train, y_test):
    EXPERIMENT_NAME = "AdaBoostClassifier - Experiment"
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    # current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    # EXPERIMENT_ID=current_experiment['experiment_id']

    # Elements of experience
    learning_rate = [random.uniform(0.1, 1.5) for _ in range(30)]
    n_estimators = [random.randint(100,200) for _ in range(10)]

    parm_list = []
    for lr in learning_rate:
      for e in n_estimators:
            parm_list.append([lr, e])

    PARAMS = {}
    MODELS = {}
    for i in range(len(parm_list)):
        PARAMS[i+1] = {"learning_rate" : parm_list[i][0], "n_estimators" : parm_list[i][1]}
        MODELS[i+1] = {"model": AdaBoostClassifier(random_state=0, learning_rate = parm_list[i][0], n_estimators = parm_list[i][1])}

    print(f"***\t***\tSTART {EXPERIMENT_NAME}\t***\t***")
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        pars = [(MODELS[i]['model'], X_train, X_test, y_train, y_test, PARAMS, i, EXPERIMENT_ID) for i in range(1, len(PARAMS)+1)]

        futures = executor.map(train, *zip(*pars))
    end = time.time()

    print(f"***\t***\tFINISH {EXPERIMENT_NAME}\t***\t***")
    
    run_time = end - start

    hours = int(run_time // 3600)
    minutes = int((run_time % 3600) // 60)
    seconds = int(run_time % 60)

    print(f"Run time execution : {hours}:{minutes}:{seconds}")