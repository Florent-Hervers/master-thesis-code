import numpy as np
import time

from dataset import SNPmarkersDataset
from sklearn.ensemble import RandomForestRegressor


def main():
    """ 
    Train an unconstrained random forest models to evaluate the maximum and average tree depth
    """
    train_dataset = SNPmarkersDataset(mode="train")
    phenotypes = list(train_dataset.phenotypes.keys())

    # These two lines are only used to compute the number of features of the training dataset for the max_features sqrt value
    train_dataset.set_phenotypes = phenotypes[0]
    X_train = train_dataset.get_all_SNP()

    for phenotype in phenotypes:
        start_time = time.time()

        train_dataset.set_phenotypes = phenotype

        X_train = train_dataset.get_all_SNP()
        Y_train = np.array(train_dataset.phenotypes[phenotype]).ravel() / train_dataset.pheno_std[phenotype]

        model = RandomForestRegressor(n_estimators=1000, random_state=2307, n_jobs=-1).fit(X_train, Y_train)

        max_depth = []
        for tree in model.estimators_:
            max_depth.append(tree.tree_.max_depth)
        
        print(f"Average tree depth: {np.array(max_depth).mean()}")
        print(f"Maximal tree depth: {max(max_depth)}")

        print("////////////////////////////////////////////")
        print(f"Computation finished in {int((time.time() - start_time) // 3600)}h {int(((time.time() - start_time) % 3600) // 60)}m {int((time.time() - start_time) % 60)}s")

if __name__ == "__main__":
    main()