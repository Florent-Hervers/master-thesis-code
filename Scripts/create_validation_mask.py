import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Specify the size of the validation (ie the number of sample to reserve for validation at the tail of the train data)
    VALIDATION_SIZE = 1000
    
    df = pd.read_csv("../Data/BBBDL_pheno_2023bbb_0twins_6traits_mask_processed.csv", index_col = 1)

    # Mask the 1000 last elements for each phenotype
    for pheno in df.drop("col_1", axis = 1).columns:
        index = df[pheno].dropna().tail(VALIDATION_SIZE).index
        for i, _ in df.iterrows():
            if i in index:
                df.loc[i, pheno] = np.nan
    print("Show resulting dataframe to check the results (only pheno 1 to 4 before BBB2024_11330 should be not nan for VALIDATION_SIZE = 1000):")
    print(df.iloc[13325:13335])

    # Reset the index and reorder the columns correctly
    df.reset_index(inplace= True)
    order = df.columns.to_list()
    tmp = order[0]
    order[0] = order[1]
    order[1] = tmp
    df.to_csv("../Data/BBBDL_pheno_2023bbb_0twins_6traits_validation_mask", sep = "\t", header=False, columns=order, index=False, na_rep= "NA")