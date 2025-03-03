import pandas as pd
import tqdm
from bed_reader import open_bed

if __name__ == "__main__":
    """Create a csv compatible file where the data is tokenize by the given size (TOKEN_SIZE).
    The variable BED_FILE_PATH must be the path to the source bed file and the output will be stored in OUTPUT_FILE_PATH.

    Raises:
        Exception: If one of the genotype has an unknown value
    """
    BED_FILE_PATH = "../Data/BBBDL_BBB2023_MD.bed"
    TOKEN_SIZE = 4
    OUTPUT_FILE_PATH = f"../Data/tokenized_genotype_{TOKEN_SIZE}.csv"

    bed = open_bed(BED_FILE_PATH, num_threads= -1)
    
    SNP_data = bed.read(num_threads=8)

    results = []
    start_index = 0
    for k, sample in tqdm.tqdm(enumerate(SNP_data)):
        def convert_bed_to_index(tuple_data):
            values = {
                "A": 0,
                "T": 1,
                "C": 2,
                "G": 3,
                "X": 4,  
            }

            index, data = tuple_data
            
            if data == 0:
                return values[bed.allele_1[index]]
            elif data == 2:
                return values[bed.allele_2[index]]
            elif data == 1:
                return values['X']
            else:
                raise Exception(f"Unknown genotype value detected: got {data} but expected 0, 1 or 2!")

        converted_sample = list(map(convert_bed_to_index, enumerate(sample)))

        tokens = []
        for i in range(0,len(converted_sample), TOKEN_SIZE):
            sum = 0
            for j in range(TOKEN_SIZE):
                sum += converted_sample[i + j] * (5 ** j)
            
            tokens.append(sum)
        results.append(tokens)

        if k % 1000 == 0 and k != 0:
            pd.DataFrame(results, index= bed.iid[start_index:k + 1]).to_csv(OUTPUT_FILE_PATH, mode = 'a')
            results = []
            start_index = k + 1

    # Add last sequences not added yet
    pd.DataFrame(results, index= bed.iid[start_index:]).to_csv(OUTPUT_FILE_PATH, mode = 'a')

    # Clean the resulting file (remove the index row for a only big one)
    # Note: if the process is killed due to a lack of memory, we can always resume from here as all relevant data is present in the file.
    print("Cleaning output file .....")
    df = pd.read_csv(OUTPUT_FILE_PATH, index_col = 0)
    df.loc[df.index.dropna("any")].to_csv(OUTPUT_FILE_PATH)