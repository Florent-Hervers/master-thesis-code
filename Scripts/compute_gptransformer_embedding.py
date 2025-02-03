import json
import numpy as np

from argparse import ArgumentParser
from bed_reader import open_bed

def compute_gptransformer_embedding_data(data_path, output_path):
    """ Generate the p and q (see paper https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.761402/full) \
        value of for every marker of the given bed file. Output the results in a json file to avoid redundant computations

    Args:
        data_path (str): Path to the bed file to process
        output_path (str): Path of the file generating the output (must include the filename and extension). 
        Every markers index is a key. For every key, a dictonary with keys ("p", "q") is available with corresponding values.
    """
    bed_file_data = np.array(open_bed(data_path).read(dtype="int8", num_threads= 8))

    results = [0] * bed_file_data.shape[1]
    for i in range(bed_file_data.shape[1]):
        _ , counts = np.unique(bed_file_data[:,i], return_counts= True)
        D = counts[0]
        H = counts[1]
        N = sum(counts)

        p = (2 * D + H) / (2 * N) 
        q = 1 - p

        results[i] = (i , {"p": p, "q": q})

    with open(output_path, "w") as f:
        json.dump(dict(results), f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="../Data/BBBDL_BBB2023_MD.bed",
        help="Path to the bed file to process"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="gptranformer_embedding_data.json",
        help="Path of the file generating the output (must include the filename and extension)"
    )
    args = parser.parse_args()
    compute_gptransformer_embedding_data(args.data_path, args.output_path)
    