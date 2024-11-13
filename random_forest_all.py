from dataset_new import SNPmarkersDataset
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_absolute_error
import time
import pandas as pd

def main():
    selected_phenotypes = ["pheno_1", "pheno_2", "pheno_3", "pheno_4"]

    train_dataset = SNPmarkersDataset(mode="train")
    train_dataset.set_phenotypes = selected_phenotypes
    validation_dataset = SNPmarkersDataset(mode="validation")
    validation_dataset.set_phenotypes = selected_phenotypes

    X_train = train_dataset.get_all_SNP()
    Y_train = pd.DataFrame([train_dataset.phenotypes[pheno] for pheno in selected_phenotypes]).transpose()
    
    X_validation = validation_dataset.get_all_SNP()
    Y_validation = pd.DataFrame([validation_dataset.phenotypes[pheno] for pheno in selected_phenotypes]).transpose()

    max_depth = [15,17,19,21,23,25,27,29,31,33,35]
    min_sample_split = [2, 4, 6, 8, 10, 12 ,14 ,16 ,18 ,20]
    nb_phenotypes = Y_validation.shape[-1]
    MAE_results = np.zeros((nb_phenotypes, len(max_depth), len(min_sample_split)))
    correlation_results = np.zeros((nb_phenotypes, len(max_depth), len(min_sample_split)))
    
    start_time = time.time()

    for i,max_depth_value in enumerate(max_depth):
        for j, sample_split in enumerate(min_sample_split):
            model = RandomForestRegressor(max_depth=max_depth_value, min_samples_split=sample_split, random_state=2307, n_jobs=-1).fit(X_train, Y_train)
            predictions = model.predict(X_validation)
            print("////////////////////////////////////////////")
            print(f"Iteration {i * len(min_sample_split) + (j+1)}/{len(min_sample_split) * len(max_depth)}")
            print(f"Max depth value tested: {max_depth_value}, min_sample_split tested: {sample_split}")
            print(f"Elapsed time from start: {int((time.time() - start_time) // 60)}m {int((time.time() - start_time) % 60)}s")
            
            for k in range(nb_phenotypes):
                MAE_results[k][i][j] = mean_absolute_error(Y_validation.iloc[:, k], predictions[:, k])
                correlation_results[k][i][j] = pearsonr(Y_validation.iloc[:, k], predictions[:, k]).statistic
                print("--------------------------------------------")
                print(f"Pearson correlation for pheno_{k + 1}: {correlation_results[k][i][j]:.5f}")
                print(f"MAE results for pheno_{k + 1}: {MAE_results[k][i][j]:.5f}")

    for k in range(nb_phenotypes):
        pd.DataFrame(MAE_results[k], 
                     index=[f"max_depth = {i}" for i in max_depth], 
                     columns=[f"min_sample_split = {i}" for i in min_sample_split]
                     ).to_csv(f"Results/random_forest_all_results_MAE_pheno_{k + 1}.csv")
        pd.DataFrame(correlation_results[k], 
                     index=[f"max_depth = {i}" for i in max_depth], 
                     columns=[f"min_sample_split = {i}" for i in min_sample_split]
                     ).to_csv(f"Results/random_forest_all_results_corr_pheno_{k + 1}.csv")
    
    print("////////////////////////////////////////////")
    print(f"Computation finished in {int((time.time() - start_time) // 3600)}h {int(((time.time() - start_time) % 3600) // 60)}m {int((time.time() - start_time) % 60)}s")

if __name__ == "__main__":
    main()