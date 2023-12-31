import sys
sys.path.append(r'D:\WORK\Personnel\Python projects\GitHub projects\Bank-Customers-Churn')
import time
import random
import concurrent.futures
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.utils import train

def KNeighborsClassifier_experience(X_train, X_test, y_train, y_test):
    EXPERIMENT_NAME = "KNeighborsClassifier - Experiment"
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    # current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    # EXPERIMENT_ID=current_experiment['experiment_id']

    # Elements of experience
    n_neighbors = [random.randint(5,15) for _ in range(100)]
    weights = ["uniform", "distance"]


    parm_list = []
    for n in n_neighbors:
        for w in weights:
                parm_list.append([n, w])

    PARAMS = {}
    MODELS = {}
    for i in range(len(parm_list)):
        PARAMS[i+1] = {"n_neighbors" : parm_list[i][0], "weights" : parm_list[i][1]}
        MODELS[i+1] = {"model": KNeighborsClassifier(n_neighbors = parm_list[i][0], weights = parm_list[i][1])}

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