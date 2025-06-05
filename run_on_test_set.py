import numpy as np
import time
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from argparse import ArgumentParser
from dataset import SNPmarkersDataset
from utils import print_elapsed_time
from xgboost import XGBRegressor


def main():
    """ 
    Evaluate the models trained with the sklearn interface on the test set with their best hyperparameters.
    For XGBoost and Random Forest, performs five runs with different seed to evaluate the variance of the model.
    """

    NB_RUNS = 5
    suppported_models = [
        "Ridge",
        "XGBoost",
        "XGBoost_all",
        "Random_forest",
        "Random_forest_all",
        "SVM",
    ]

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, choices=suppported_models, help="Model on which we should train and evaluate on the test set.")

    model_name = parser.parse_args().model

    train_dataset = SNPmarkersDataset(mode="train")
    test_dataset = SNPmarkersDataset(mode="test")

    if model_name == "XGBoost_all" or model_name == "Random_forest_all":
        phenotypes = [["ep_res", "de_res", "FESSEp_res", "FESSEa_res"]]
    else:
        phenotypes = list(train_dataset.phenotypes.keys())
    
    for phenotype in phenotypes:
        train_dataset.set_phenotypes = phenotype
        test_dataset.set_phenotypes = phenotype

        X_train = train_dataset.get_all_SNP()
        
        if model_name == "XGBoost_all" or model_name == "Random_forest_all":
            Y_train = pd.DataFrame([train_dataset.phenotypes[pheno] for pheno in phenotype]).transpose()
        else:
            Y_train = np.array(train_dataset.phenotypes[phenotype]).ravel()
        
        X_test = test_dataset.get_all_SNP()
        
        if model_name == "XGBoost_all" or model_name == "Random_forest_all":
            Y_test = pd.DataFrame([test_dataset.phenotypes[pheno] for pheno in phenotype]).transpose()
        else:
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
            
            elif model_name == "Random_forest_all":
                hp = {"max_feature": 0.1, "max_depth": 17}
                model = RandomForestRegressor(n_estimators=1000,
                                              max_depth=hp["max_depth"],
                                              max_features=hp["max_feature"],
                                              random_state=2307 + i,
                                              n_jobs=-1)
            
            elif model_name == "XGBoost_all":
                hp = {"max_depth": 7, "learning_rate": 0.01, "subsampling": 0.75, "normalization": True}
                model = XGBRegressor(n_estimators=1000,
                                     subsample= hp["subsampling"],
                                     learning_rate=hp["learning_rate"],
                                     max_depth= hp["max_depth"],
                                     n_jobs = -1,
                                     random_state=2307 + i)
            
            elif model_name == "XGBoost":
                hp = {
                    "ep_res": {"max_depth": 7, "learning_rate": 0.01, "subsampling": 0.75},
                    "de_res": {"max_depth": 7, "learning_rate": 0.01, "subsampling": 0.75},
                    "FESSEp_res": {"max_depth": 4, "learning_rate": 0.1, "subsampling": 0.75},
                    "FESSEa_res": {"max_depth": 3, "learning_rate": 0.1, "subsampling": 0.75},
                    "size_res": {"max_depth": 9, "learning_rate": 0.01, "subsampling": 0.25},
                    "MUSC_res": {"max_depth": 3, "learning_rate": 0.1, "subsampling": 0.75},
                }
                model = XGBRegressor(n_estimators=1000,
                                     subsample= hp[phenotype]["subsampling"],
                                     learning_rate=hp[phenotype]["learning_rate"],
                                     max_depth= hp[phenotype]["max_depth"],
                                     n_jobs = -1,
                                     random_state=2307 + i)

            elif model_name == "SVM":
                # There is no point to run several times this model
                if i >= 1:
                    break
                hp = {
                    "ep_res": {"C": 3.25, "gamma": 1.225e-4},
                    "de_res": {"C": 7.25, "gamma": 1.225e-4},
                    "FESSEp_res": {"C": 3.25, "gamma": 7.75e-5},
                    "FESSEa_res": {"C": 5.5, "gamma": 1e-4},
                    "size_res": {"C": 21.25, "gamma": 1e-4},
                    "MUSC_res": {"C": 3.25, "gamma": 1e-5},
                }
                model = SVR(gamma=hp[phenotype]["gamma"], C=hp[phenotype]["C"])

            # Perform only one time the normalization if necessary
            if i == 0:
                if "normalization" in hp.keys() and hp["normalization"]:
                    for pheno in Y_train:
                        Y_train[pheno] /= train_dataset.pheno_std[pheno]

            model = model.fit(X_train, Y_train)
            
            predictions = model.predict(X_test)
            

            if model_name == "XGBoost_all" or model_name == "Random_forest_all":
                
                tmp_MAE = []
                tmp_corr = []
                for i, pheno in enumerate(phenotype):
                    if "normalization" in hp.keys() and hp["normalization"]:
                        tmp_MAE.append(mean_absolute_error(Y_test.iloc[:, i], predictions[:, i] * test_dataset.pheno_std[pheno]))
                    else:
                        tmp_MAE.append(mean_absolute_error(Y_test.iloc[:, i] , predictions[:, i]))
                    tmp_corr.append(pearsonr(Y_test.iloc[:, i], predictions[:, i]).statistic)
                MAE.append(tmp_MAE)
                correlation.append(tmp_corr)

            else:
                MAE.append(mean_absolute_error(Y_test, predictions))
                correlation.append(pearsonr(Y_test, predictions).statistic)

        # This condition enable to skip the printing when the hyperparmater to use weren't defined yet
        if len(correlation) > 0 and len(MAE) > 0:
            print("////////////////////////////////////////////")
            print(f"Evaluation of model {model_name} finshed for phenotype {phenotype} in {print_elapsed_time(start_time)}")
            print(f"Hyper parameters tested: {hp if type(phenotype) == list else hp[phenotype]}")
            print(f"Results:")
            print(f"    -> Correlation : {correlation}")
            print(f"        - mean: {np.array(correlation).mean(axis = 0)}")
            print(f"        - std: {np.array(correlation).std(axis = 0)}")
            print(f"    -> MAE : {MAE}")
            print(f"        - mean: {np.array(MAE).mean(axis = 0)}")
            print(f"        - std: {np.array(MAE).std(axis = 0)}")
        

if __name__ == "__main__":
    main()