import pandas as pd
import tqdm
from bed_reader import open_bed
from argparse import ArgumentParser

if __name__ == "__main__":
    """Create a csv compatible file where the data is tokenize by the given size (token_size argument).
    The bed_path argument must be the path to the source bed file and the output will be stored in {output_path}_{vocab_size}_{token_size}.csv.

    Raises:
        Exception: If an error occured, one of the genotype has an unknown value or the sequences isn't divisible by the token size or the vocab size isn't supported.
    """
    parser = ArgumentParser()

    parser.add_argument("--bed_path", "-b", default="../Data/BBBDL_BBB2023_MD.bed", type=str, help="Relative filepath to the bed file")
    parser.add_argument("--token_size", "-t", required=True, type=int, help= "Size of the tokens to creates (the sequence length should be divisible by this value)")
    parser.add_argument("--output_path", "-o", default="../Data/tokenized_genotype_enhanced", type=str, help="Relative path + filename prefix of the ouput of the script")
    parser.add_argument("--vocab_size", "-v", required=True, type=int, help="Size of the vocabulary that define the encoding to use")

    args = parser.parse_args()

    OUTPUT_FILE_PATH = f"{args.output_path}_{args.vocab_size}_{args.token_size}.csv"

    try:
        bed = open_bed(args.bed_path, num_threads= -1)
    except Exception as e:
        raise Exception(f"The following error occured during the opening of the bed file: {e.args}")

    SNP_data = bed.read(num_threads=8)

    if len(SNP_data[0]) % args.token_size != 0:
        raise Exception(f"The sequence size ({len(SNP_data[0])}) must be divisible by the token size ({args.token_size})")
    
    SUPPORTED_VOCAB_SIZE = [3,5,9]
    if args.vocab_size not in SUPPORTED_VOCAB_SIZE:
        raise Exception(f"The given vocab size ({args.vocab_size}) isn't supported. Supported vocab size are {SUPPORTED_VOCAB_SIZE}")

    results = []
    start_index = 0
    for k, sample in tqdm.tqdm(enumerate(SNP_data)):
        def convert_bed_to_index(tuple_data):
            """Convert the addivitive encoding into the index to use for the tokenization

            Args:
                tuple_data (tuple): first element should be the position (starting at 0) in the sequence while the second value is the 0/1/2 value.

            Raises:
                Exception: If the second value of the tuple isn't in [0,1,2]

            Returns:
                int: Index to be used to create the tokens
            """
            # Every value of the variable SUPPORTED_VOCAB_SIZE should be handeled in this function
            if args.vocab_size == 9:
                values = {
                    "A": 0,
                    "T": 1,
                    "C": 2,
                    "G": 3,
                    "X": 4, 
                    "a": 5,
                    "t": 6,
                    "c": 7,
                    "g": 8, 
                }
            elif args.vocab_size == 5:
                values = {
                    "A": 0,
                    "T": 1,
                    "C": 2,
                    "G": 3,
                    "X": 4, 
                }
            elif args.vocab_size == 3:
                return tuple_data[1]
            
            index, data = tuple_data
            
            if data == 0:
                return values[bed.allele_1[index]]
            elif data == 2:
                if args.vocab_size == 9:
                    return values[bed.allele_2[index].lower()]
                else:
                    return values[bed.allele_2[index]]
            elif data == 1:
                return values['X']
            else:
                raise Exception(f"Unknown genotype value detected: got {data} but expected 0, 1 or 2!")

        converted_sample = list(map(convert_bed_to_index, enumerate(sample)))

        tokens = []
        for i in range(0,len(converted_sample), args.token_size):
            sum = 0
            for j in range(args.token_size):
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
    print(f"Creation of the tokens finished! The tokenized sequences can be found at {OUTPUT_FILE_PATH}")