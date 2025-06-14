import time
import json
import numpy as np

from dataset import SNPmarkersDataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from utils import print_elapsed_time

def main():
    """ 
    Trained the ridge regression model on the phenotypes of the phenotype variable with 
    the hardcoded range of hyperparameters in the variables lambdas.
    The results are stored in a json file, the keys of the json object are: 
    
    - dim_0_values: contain the values of lamdba tested.
    - dim_0_label: contain the name of the hyperparamter to ease the plotting of the results.
    - correlation: array of the same size than dim_0_values that contains 
        at index i the correlation on the validation set for the model that uses
        the value of lambda at index i of dim_0_values.
    - MAE: array of the same size than dim_0_values that contains 
        at index i the mean average error on the validation set for the model that uses
        the value of lambda at index i of dim_0_values.
    """
    train_dataset = SNPmarkersDataset(mode="train")
    validation_dataset = SNPmarkersDataset(mode="validation")
    phenotypes = ["ep_res"]
    
    lambdas = np.linspace(46050, 70000, 480)

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

        max_correlation = 0
        early_stop_counter = 0
        EARLY_STOP_THRESHOLD = 30

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

            if correlation_results[i] >= max_correlation:
                max_correlation = correlation_results[i]
                early_stop_counter = 0
            else:
                if early_stop_counter >= EARLY_STOP_THRESHOLD:
                    break
                else:
                    early_stop_counter += 1

        print("////////////////////////////////////////////")
        print(f"Computation finished for {pheno} in {print_elapsed_time(start_time)}")

        with open(f"Results/ridge_16_{pheno}.json", "w") as f:
            results = {
                "dim_0_values": lambdas.tolist()[0:iteration_counter],
                "dim_0_label": "lambda",
                "correlation": correlation_results.tolist()[0:iteration_counter],
                "MAE": MAE_results.tolist()[0:iteration_counter]
            }
            json.dump(results, f)
            
if __name__ == "__main__":
    main()