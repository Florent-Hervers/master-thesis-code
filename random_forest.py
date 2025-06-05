from dataset import SNPmarkersDataset
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_absolute_error
import time
import pandas as pd
from utils import print_elapsed_time

def main():
    """ 
    Trained the random forest algorithm on the phenotypes of the phenotype variable with 
    the hardcoded range of hyperparameters in the variables max_depth and max_features.
    The results are stored in pandas Dataframes with the value of max depth as index and the max features as columns.
    """
    train_dataset = SNPmarkersDataset(mode="train")
    validation_dataset = SNPmarkersDataset(mode="validation")
    phenotypes = list(train_dataset.phenotypes.keys())

    # These two lines are only used to compute the number of features of the training dataset for the max_features sqrt value
    train_dataset.set_phenotypes = phenotypes[0]
    X_train = train_dataset.get_all_SNP()

    max_depth = np.arange(5,19)
    max_features = [0.5, 1/3, 0.25, 0.1, 0.01, int(np.sqrt(X_train.shape[-1])), 0.001]
    MAE_results = np.zeros((len(max_depth), len(max_features)))
    correlation_results = np.zeros((len(max_depth), len(max_features)))
    
    for phenotype in phenotypes:
        start_time = time.time()

        train_dataset.set_phenotypes = phenotype
        validation_dataset.set_phenotypes = phenotype

        X_train = train_dataset.get_all_SNP()
        Y_train = np.array(train_dataset.phenotypes[phenotype]).ravel()
        Y_train /= train_dataset.pheno_std[phenotype]

        X_validation = validation_dataset.get_all_SNP()
        Y_validation = np.array(validation_dataset.phenotypes[phenotype]).ravel()
        Y_validation /= validation_dataset.pheno_std[phenotype]

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
                
                MAE_results[i][j] = mean_absolute_error(Y_validation * validation_dataset.pheno_std[phenotype], predictions * validation_dataset.pheno_std[phenotype])
                correlation_results[i][j] = pearsonr(Y_validation, predictions).statistic
                print("--------------------------------------------")
                print(f"Pearson correlation for {phenotype}: {correlation_results[i][j]:.5f}")
                print(f"MAE results for {phenotype}: {MAE_results[i][j]:.5f}")

        pd.DataFrame(MAE_results, 
                    index=[f"max_depth = {i}" for i in max_depth], 
                    columns=[f"max_features = {i}" for i in max_features]
                    ).to_csv(f"Results/random_forest_normalized_1000_results_MAE_{phenotype}.csv")
        pd.DataFrame(correlation_results, 
                    index=[f"max_depth = {i}" for i in max_depth], 
                    columns=[f"max_features = {i}" for i in max_features]
                    ).to_csv(f"Results/random_fores_normalizedt_1000_results_corr_{phenotype}.csv")
        
        print("////////////////////////////////////////////")
        print(f"Computation finished in {print_elapsed_time(start_time)}")

if __name__ == "__main__":
    main()