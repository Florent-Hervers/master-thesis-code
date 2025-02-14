import numpy as np
import time

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from argparse import ArgumentParser
from dataset import SNPmarkersDataset
from utils import print_elapsed_time

def main():
    NB_RUNS = 5

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Ridge", "Random_forest","SVM"], help="Model on which we should train and evaluate on the test set.")

    model_name = parser.parse_args().model

    train_dataset = SNPmarkersDataset(mode="train")
    test_dataset = SNPmarkersDataset(mode="test")
    phenotypes = list(train_dataset.phenotypes.keys())
    
    for phenotype in phenotypes:
        train_dataset.set_phenotypes = phenotype
        test_dataset.set_phenotypes = phenotype

        X_train = train_dataset.get_all_SNP()
        Y_train = np.array(train_dataset.phenotypes[phenotype]).ravel()
        
        X_test = test_dataset.get_all_SNP()
        Y_test = np.array(test_dataset.phenotypes[phenotype]).ravel()

        correlation = []
        MAE = []
        start_time = time.time()
        for i in range(NB_RUNS):

            if model_name == "Ridge":
                # There is no point to run several times this model
                if i >= 1:
                    break
                hp = {
                    "ep_res": {"lambda": 55600},
                    "de_res": {"lambda": 44500},
                    "FESSEp_res": {"lambda": 26250},
                    "FESSEa_res": {"lambda": 34000},
                    "size_res": {"lambda": 20900},
                    "MUSC_res": {"lambda": 23950},
                }
                model = Ridge(alpha= hp[phenotype]["lambda"], random_state= 2307 + i)
            
            elif model_name == "Random_forest":
                hp = {
                    "ep_res": {"max_feature": 0.1, "max_depth": 35},
                    "de_res": {"max_feature": 0.333333333333333333333, "max_depth": 33},
                    "FESSEp_res": {"max_feature": 0.01, "max_depth": 17},
                    "FESSEa_res": {"max_feature": 0.25, "max_depth": 23},
                    "size_res": {"max_feature": 0.1, "max_depth": 25},
                    "MUSC_res": {"max_feature": 0.25, "max_depth": 15},
                }
                model = RandomForestRegressor(n_estimators=1000,
                                              max_depth=hp[phenotype]["max_depth"],
                                              max_features=hp[phenotype]["max_feature"],
                                              random_state=2307 + i,
                                              n_jobs=-1)

            elif model_name == "SVM":
                # Skip others phenotypes as hyperparameters aren't defined yet
                if phenotype != "ep_res":
                    break
                
                # There is no point to run several times this model
                if i >= 1:
                    break
                hp = {
                    "ep_res": {"C": 3.25, "gamma": 1.225e-4},
                    "de_res": {"C": 0, "gamma": 0},
                    "FESSEp_res": {"C": 0, "gamma": 0},
                    "FESSEa_res": {"C": 0, "gamma": 0},
                    "size_res": {"C": 0, "gamma": 0},
                    "MUSC_res": {"C": 0, "gamma": 0},
                }
                model = SVR(gamma=hp[phenotype]["gamma"], C=hp[phenotype]["C"])

            model = model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            
            MAE.append(mean_absolute_error(Y_test , predictions))
            correlation.append(pearsonr(Y_test, predictions).statistic)

        # This condition enable to skip the printing when the hyperparmater to use weren't defined yet
        if len(correlation) > 0 and len(MAE) > 0:
            print("////////////////////////////////////////////")
            print(f"Evaluation of model {model_name} finshed for phenotype {phenotype} in {print_elapsed_time(start_time)}")
            print(f"Hyper parameters tested: {hp[phenotype]}")
            print(f"Results:")
            print(f"    -> Correlation : {correlation}")
            print(f"        - mean: {np.array(correlation).mean()}")
            print(f"        - std: {np.array(correlation).std()}")
            print(f"    -> MAE : {MAE}")
            print(f"        - mean: {np.array(MAE).mean()}")
            print(f"        - std: {np.array(MAE).std()}")
        

if __name__ == "__main__":
    main()