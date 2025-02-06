import time
import json
import numpy as np

from dataset import SNPmarkersDataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from utils import print_elapsed_time

def main():
    train_dataset = SNPmarkersDataset(mode="train")
    validation_dataset = SNPmarkersDataset(mode="validation")
    phenotypes = list(train_dataset.phenotypes)
    
    lambdas = np.linspace(19050, 22000, 60)

    MAE_results = np.zeros((len(lambdas)))
    correlation_results = np.zeros((len(lambdas)))

    for pheno in phenotypes:
        start_time = time.time()
        iteration_counter = 0

        train_dataset.set_phenotypes = pheno
        validation_dataset.set_phenotypes = pheno

        X_train = train_dataset.get_all_SNP()
        Y_train = np.array(train_dataset.phenotypes[pheno]).ravel()
        
        X_validation = validation_dataset.get_all_SNP()
        Y_validation = np.array(validation_dataset.phenotypes[pheno]).ravel()

        for i,lambda_val in enumerate(lambdas):
            model = Ridge(
                alpha= lambda_val,
                random_state= 2307,
            )
            model = model.fit(X_train, Y_train)
            validation_predictions = model.predict(X_validation)
            
            MAE_results[i] = mean_absolute_error(Y_validation , validation_predictions) 
            correlation_results[i] = pearsonr(Y_validation, validation_predictions).statistic
                
            iteration_counter += 1
            
            print("////////////////////////////////////////////")
            print(f"Iteration {iteration_counter}/{len(lambdas)} for phenotype {pheno} finished")
            print(f"Lambda tested : {lambda_val}")
            print(f"Elapsed time from start: {print_elapsed_time(start_time)}")
            print(f"Results:")
            print(f"    - MAE : {MAE_results[i]}")
            print(f"    - Correlation : {correlation_results[i]}")

        print("////////////////////////////////////////////")
        print(f"Computation finished for {pheno} in {print_elapsed_time(start_time)}")

        with open(f"Results/ridge_11_{pheno}.json", "w") as f:
            results = {
                "dim_0_values": lambdas.tolist(),
                "dim_0_label": "lambda",
                "correlation": correlation_results.tolist(),
                "MAE": MAE_results.tolist()
            }
            json.dump(results, f)
            
if __name__ == "__main__":
    main()