import time
import json
import cupy as cp
import numpy as np

from dataset import SNPmarkersDataset
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from utils import print_elapsed_time

def main():
    """ 
    Trained the ridge regression model on the phenotypes of the phenotype variable with 
    the hardcoded range of hyperparameters in the variables sub_sampling, learning_rates, and max_depth.
    The results are stored in a json file, the keys of the json object are: 
    
    - dim_0_values: contain the values for the subsampling hyperparameter.
    - dim_0_label: contain the name of the hyperparamter to ease the plotting of the results.
    - dim_1_values: contain the values for the learning_rate hyperparameter.
    - dim_1_label: contain the name of the hyperparamter to ease the plotting of the results.
    - dim_2_values: contain the values for the max_depth hyperparameter.
    - dim_2_label: contain the name of the hyperparamter to ease the plotting of the results.
    - correlation: 3D multi-dimentional array that contains 
        at index i, j, k the correlation on the validation set for the model that uses:
        - the sub_sampling at index i of dim_0_values.
        - the learning rate at index j of dim_1_values.
        - the max_depth at index k of dim_2_values.
    - MAE: 3D multi-dimentional array that contains 
        at index i, j, k the mean average error on the validation set for the model that uses:
        - the sub_sampling at index i of dim_0_values. 
        - the learning rate at index j of dim_1_values.
        - the max_depth at index k of dim_2_values.
    """
    NORMALIZATION = True

    train_dataset = SNPmarkersDataset(mode="train")
    validation_dataset = SNPmarkersDataset(mode="validation")
    phenotypes = list(train_dataset.phenotypes.keys())
    
    sub_sampling = [0.75, 0.25]
    learning_rates = [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0005, 0.0001]
    max_depth = [10, 9, 8, 7, 6, 5, 4, 3, 2]

    MAE_results = np.zeros((len(sub_sampling), len(learning_rates), len(max_depth)))
    correlation_results = np.zeros((len(sub_sampling), len(learning_rates), len(max_depth)))

    for pheno in phenotypes:
        start_time = time.time()
        iteration_counter = 0
        
        train_dataset.set_phenotypes = pheno
        validation_dataset.set_phenotypes = pheno

        # Put everything on the GPU speed thing up
        X_train = cp.array(train_dataset.get_all_SNP())
        Y_train_cpu = np.array(train_dataset.phenotypes[pheno]).ravel()
        if NORMALIZATION:
            Y_train_cpu /= train_dataset.pheno_std[pheno]
        Y_train_gpu = cp.array(Y_train_cpu)
        
        X_validation = cp.array(validation_dataset.get_all_SNP())
        Y_validation = np.array(validation_dataset.phenotypes[pheno]).ravel()
        if NORMALIZATION:
            Y_validation /= validation_dataset.pheno_std[pheno]


        for i,sub_sampling_value in enumerate(sub_sampling):
            for j,learning_rates_value in enumerate(learning_rates):
                for k,depth in enumerate(max_depth):
                    model = XGBRegressor(n_estimators=1000,
                                        subsample=sub_sampling_value,
                                        learning_rate=learning_rates_value,
                                        max_depth= depth,
                                        n_jobs = -1,
                                        random_state=2307, 
                                        device="gpu")
                    model = model.fit(X_train, Y_train_gpu)
                    validation_predictions = model.predict(X_validation)
                    
                    if NORMALIZATION:
                        MAE_results[i,j,k] = mean_absolute_error(Y_validation * validation_dataset.pheno_std[pheno], validation_predictions* validation_dataset.pheno_std[pheno]) 
                    else:
                        MAE_results[i,j,k] = mean_absolute_error(Y_validation, validation_predictions) 
                    
                    correlation_results[i,j,k] = pearsonr(Y_validation, validation_predictions).statistic
                        
                    iteration_counter += 1
                    
                    print("////////////////////////////////////////////")
                    print(f"Iteration {iteration_counter}/{len(sub_sampling) * len(learning_rates) * len(max_depth)} for phenotype {pheno} finished")
                    print("Hyper parameters tested:")
                    print(f"    - sub_sampling: {sub_sampling_value}")
                    print(f"    - learning_rate: {learning_rates_value}")
                    print(f"    - depth: {depth}")
                    print(f"Elapsed time from start: {print_elapsed_time(start_time)}")
                    print(f"Results:")
                    print(f"    - MAE : {MAE_results[i,j,k]}")
                    print(f"    - Correlation : {correlation_results[i,j,k]}")

        print("////////////////////////////////////////////")
        print(f"Computation finished in {print_elapsed_time(start_time)}")

        with open(f"Results/xgboost_{pheno}_2{'_normalized' if NORMALIZATION else ''}_1000_results.json", "w") as f:
            results = {
                "dim_0_values": sub_sampling,
                "dim_0_label": "sub_sampling",
                "dim_1_values": learning_rates,
                "dim_1_label": "learning_rates",
                "dim_2_values": max_depth,
                "dim_2_label": "max_depth",
                "correlation": correlation_results.tolist(),
                "MAE": MAE_results.tolist()
            }
            json.dump(results, f)
            
if __name__ == "__main__":
    main()