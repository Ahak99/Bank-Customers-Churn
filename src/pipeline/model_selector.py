import sys
sys.path.append('D:\WORK\Personnel\Python projects\GitHub projects\Bank-Customers-Churn')
import pandas as pd
import concurrent.futures
import dill


from src.pipeline.data_prep import DataPrep
from src.utils import Best_in_experiment

def best_model(EXPERIMENTS):
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = executor.map(Best_in_experiment, EXPERIMENTS)
            
    experiments_infos = pd.DataFrame(columns=['Model', 'run_id','experiment_id','model_uri','Precision','F1_score'])
    for i in EXPERIMENTS:
        dic_ = dill.load(open(f"BEST_experiment\Best_{i}.pkl", "rb"))        
        dic_1 = pd.DataFrame([list(dic_.values())],columns = list(dic_.keys())) 
        experiments_infos = pd.concat([experiments_infos, dic_1], ignore_index=True)
    
    experiments_infos = experiments_infos.sort_values(by='F1_score', ascending=False)
    experiments_infos.to_csv('experiments_infos.csv', index=False)

    experiments_infos = pd.read_csv("experiments_infos.csv")
        
    print("\n\n\n\t\t\t**    **  **  Show experiments infos  **  **    **\n\n")
    print(experiments_infos.head(len(EXPERIMENTS)))
    print("\n\n\n")
        
    best_predictor = experiments_infos.iloc[0]["model_uri"]
    
    return best_predictor