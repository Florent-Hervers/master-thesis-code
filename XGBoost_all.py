from dataset import SNPmarkersDataset
from xgboost import XGBRegressor, set_config
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import json
import cupy as cp

def main():
    train_dataset = SNPmarkersDataset(mode="train")
    validation_dataset = SNPmarkersDataset(mode="validation")

    # put everything on the GPU speed thing up
    X_train = cp.array(train_dataset.input)
    Y_train_gpu = cp.array(train_dataset.pheno)
    Y_train_cpu = train_dataset.pheno

    X_validation = cp.array(validation_dataset.input)
    Y_validation = validation_dataset.pheno
    
    # For the three first hyper parameters, the first one is the default one.
    sub_sampling = [1, 0.5]
    learning_rates = [0.3, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5 ]
    min_sample_split = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    n_estimators = np.arange(1,1000,2).tolist()


    nb_phenotypes = Y_validation.shape[-1]
    train_loss = np.zeros((nb_phenotypes, len(sub_sampling), len(learning_rates), len(min_sample_split)))
    validation_loss = np.zeros((nb_phenotypes, len(sub_sampling), len(learning_rates), len(min_sample_split)))
    MAE_results = np.zeros((nb_phenotypes, len(sub_sampling), len(learning_rates), len(min_sample_split)))
    correlation_results = np.zeros((nb_phenotypes, len(sub_sampling), len(learning_rates), len(min_sample_split)))
    
    start_time = time.time()
    iteration_counter = 0
    for i,sub_sampling_value in enumerate(sub_sampling):
        for j,learning_rates_value in enumerate(learning_rates):
            for k,min_sample_split_value in enumerate(min_sample_split):
                #for l, n_estimators_value in enumerate(n_estimators):
                    model = XGBRegressor(subsample=sub_sampling_value,
                                         learning_rate=learning_rates_value,
                                         min_child_weight=min_sample_split_value,
                                         n_jobs = -1,
                                         random_state=2307, 
                                         device="gpu")
                    model = model.fit(X_train, Y_train_gpu)
                    train_predictions = model.predict(X_train)
                    validation_predictions = model.predict(X_validation)

                    for m in range(nb_phenotypes):
                        train_loss[m,i,j,k] = mean_squared_error(Y_train_cpu.iloc[:, m], train_predictions[:, m])
                        validation_loss[m,i,j,k] = mean_squared_error(Y_validation.iloc[:, m], validation_predictions[:, m])
                        MAE_results[m,i,j,k] = mean_absolute_error(Y_validation.iloc[:, m], validation_predictions[:, m])
                        correlation_results[m,i,j,k] = pearsonr(Y_validation.iloc[:, m], validation_predictions[:, m]).statistic
            
            iteration_counter += 1
            print("////////////////////////////////////////////")
            print(f"Iteration {iteration_counter}/{len(sub_sampling) * len(learning_rates) * len(min_sample_split)} finished")
            print("Hyper parameters tested:")
            print(f"    - sub_sampling: {sub_sampling_value}")
            print(f"    - learning_rate: {learning_rates_value}")
            print(f"    - min_sample_split: {min_sample_split_value}")
            print(f"Elapsed time from start: {int((time.time() - start_time) // 60)}m {int((time.time() - start_time) % 60)}s")

    print("////////////////////////////////////////////")
    print(f"Computation finished in {int((time.time() - start_time) // 3600)}h {int(((time.time() - start_time) % 3600) // 60)}m {int((time.time() - start_time) % 60)}s")

    with open("Results/xgboost_all_results.json", "w") as f:
        results = {
            "dim_0_values": Y_validation.columns.to_list(),
            "dim_0_label": "phenotypes",
            "dim_1_values": sub_sampling,
            "dim_1_label": "sub_sampling",
            "dim_2_values": learning_rates,
            "dim_2_label": "learning_rates",
            "dim_3_values": min_sample_split,
            "dim_3_label": "min_sample_split",
            "train_loss": train_loss.tolist(),
            "validation_loss": validation_loss.tolist(),
            "correlation": correlation_results.tolist(),
            "MAE": MAE_results.tolist()
        }
        json.dump(results, f)

if __name__ == "__main__":
    main()