import time
import pandas as pd
import numpy as np

from dataset import SNPmarkersDataset
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from utils import print_elapsed_time

def main():
    """ 
    Trained the random forest algorithm on the multi-trait regression problem with 
    the hardcoded range of hyperparameters in the variables max_depth and max_features.
    The results are stored in pandas Dataframes with the value of max depth as index and the max features as columns.
    """
    selected_phenotypes = ["ep_res","de_res","FESSEp_res","FESSEa_res"]

    train_dataset = SNPmarkersDataset(mode="train")
    train_dataset.set_phenotypes = selected_phenotypes
    validation_dataset = SNPmarkersDataset(mode="validation")
    validation_dataset.set_phenotypes = selected_phenotypes

    X_train = train_dataset.get_all_SNP()
    Y_train = pd.DataFrame([train_dataset.phenotypes[pheno] for pheno in selected_phenotypes]).transpose()
    for pheno in Y_train:
        Y_train[pheno] /= train_dataset.pheno_std[pheno]
    

    X_validation = validation_dataset.get_all_SNP()
    Y_validation = pd.DataFrame([validation_dataset.phenotypes[pheno] for pheno in selected_phenotypes]).transpose()
    for pheno in Y_validation:
        Y_validation[pheno] /= validation_dataset.pheno_std[pheno]


    max_depth = np.arange(5,19)
    max_features = [0.5, 1/3, 0.25, 0.1, 0.01, int(np.sqrt(X_train.shape[-1])), 0.001]
    nb_phenotypes = len(selected_phenotypes)
    MAE_results = np.zeros((nb_phenotypes, len(max_depth), len(max_features)))
    correlation_results = np.zeros((nb_phenotypes, len(max_depth), len(max_features)))
    
    start_time = time.time()

    for i,max_depth_value in enumerate(max_depth):
        for j, max_feature in enumerate(max_features):
            model = RandomForestRegressor(n_estimators=1000, max_depth=max_depth_value, max_features=max_feature, random_state=2307, n_jobs=-1).fit(X_train, Y_train)
            predictions = model.predict(X_validation)
            print("////////////////////////////////////////////")
            print(f"Iteration {i * len(max_features) + (j+1)}/{len(max_features) * len(max_depth)}")
            if type(max_feature) == int:
                max_nb_of_tree = max_feature
            elif type(max_feature) == float:
                max_nb_of_tree = int(max_feature*X_train.shape[-1])
            print(f"Max depth value tested: {max_depth_value}, max nb of features used per tree: {max_nb_of_tree}")
            print(f"Elapsed time from start: {print_elapsed_time(start_time)}")
            
            for k in range(nb_phenotypes):
                MAE_results[k][i][j] = mean_absolute_error(Y_validation.iloc[:, k]* validation_dataset.pheno_std[selected_phenotypes[k]], predictions[:, k] * validation_dataset.pheno_std[selected_phenotypes[k]])
                correlation_results[k][i][j] = pearsonr(Y_validation.iloc[:, k], predictions[:, k]).statistic
                print("--------------------------------------------")
                print(f"Pearson correlation for {selected_phenotypes[k]}: {correlation_results[k][i][j]:.5f}")
                print(f"MAE results for {selected_phenotypes[k]}: {MAE_results[k][i][j]:.5f}")

    for k in range(nb_phenotypes):
        pd.DataFrame(MAE_results[k], 
                     index=[f"max_depth = {i}" for i in max_depth], 
                     columns=[f"max_features = {i}" for i in max_features]
                     ).to_csv(f"Results/random_forest_all_normalized_1000_results_MAE_{selected_phenotypes[k]}.csv")
        pd.DataFrame(correlation_results[k], 
                     index=[f"max_depth = {i}" for i in max_depth], 
                     columns=[f"max_features = {i}" for i in max_features]
                     ).to_csv(f"Results/random_forest_all_normalized_1000_results_corr_{selected_phenotypes[k]}.csv")
    
    print("////////////////////////////////////////////")
    print(f"Computation finished in {print_elapsed_time(start_time)}")

if __name__ == "__main__":
    main()