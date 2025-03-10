import time
import json
import numpy as np

from dataset import SNPmarkersDataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from utils import print_elapsed_time

def main():
    train_dataset = SNPmarkersDataset(mode="train", skip_check=True)
    validation_dataset = SNPmarkersDataset(mode="validation", skip_check=True)
    phenotypes = ["FESSEa_res"] #list(train_dataset.phenotypes.keys())

    gammas = np.linspace(1e-5,5.5e-5,3)
    c = np.concatenate([np.array([0.9]), np.linspace(1.0, 7.75, 4)])

    MAE_results = np.zeros((len(gammas), len(c)))
    correlation_results = np.zeros((len(gammas), len(c)))
    for pheno in phenotypes:
        start_time = time.time()
        iteration_counter = 0

        train_dataset.set_phenotypes = pheno
        validation_dataset.set_phenotypes = pheno

        X_train = train_dataset.get_all_SNP()
        Y_train = np.array(train_dataset.phenotypes[pheno]).ravel()
        
        X_validation = validation_dataset.get_all_SNP()
        Y_validation = np.array(validation_dataset.phenotypes[pheno]).ravel()

        for i,gamma_val in enumerate(gammas):
            for j, c_val in enumerate(c):
                model = SVR(
                    gamma=gamma_val,
                    C= c_val,
                )
                model = model.fit(X_train, Y_train)
                validation_predictions = model.predict(X_validation)
                
                MAE_results[i,j] = mean_absolute_error(Y_validation , validation_predictions) 
                correlation_results[i,j] = pearsonr(Y_validation, validation_predictions).statistic
                    
                iteration_counter += 1
                
                print("////////////////////////////////////////////")
                print(f"Iteration {iteration_counter}/{len(gammas) * len(c)} for phenotype {pheno} finished")
                print("Hyper parameters tested:")
                print(f"    - gamma: {gamma_val}")
                print(f"    - c: {c_val}")
                print(f"Elapsed time from start: {print_elapsed_time(start_time)}")
                print(f"Results:")
                print(f"    - MAE : {MAE_results[i,j]}")
                print(f"    - Correlation : {correlation_results[i,j]}")

        print("////////////////////////////////////////////")
        print(f"Computation finished in {print_elapsed_time(start_time)}")

        with open(f"Results/SVM_14_{pheno}.json", "w") as f:
            results = {
                "dim_0_values": gammas.tolist(),
                "dim_0_label": "gamma",
                "dim_1_values": c.tolist(),
                "dim_1_label": "c",
                "correlation": correlation_results.tolist(),
                "MAE": MAE_results.tolist()
            }
            json.dump(results, f)

if __name__ == "__main__":
    main()