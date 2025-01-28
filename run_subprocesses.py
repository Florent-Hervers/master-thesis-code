import subprocess
import time

from argparse import ArgumentParser
from utils import print_elapsed_time
from dataset import SNPmarkersDataset

def list_of_str(args):
    return args.split(",")

if __name__ == "__main__":
    """
    Create a launch a subprocess and run the given script for every available phenotype in order to speed algorithm 
    that cannot use multithreading.
    The given script should take the phenotype (type = str) as the first positional argument, additionnal arguments can
    be provided using the --arguments option.
    """

    parser = ArgumentParser()

    parser.add_argument("-s", "--script", type=str, required=True, help="The python script to run on the subprocesses (with the .py extention)")
    parser.add_argument("-a", "--arguments", type=list_of_str, default=[], help= "Arguments of the scripts to run \
                        with spaces replaced by commas (example: -p,ep_res,--epoch,100)")

    args = parser.parse_args()

    # Only use the dataset class to fetch the phenotypes to use
    phenotypes = list(SNPmarkersDataset(mode = "train", skip_check=True).phenotypes)
    processes = []
    start_time = time.time()
    for phenotype in phenotypes:
        processes.append(
            subprocess.Popen(
                ["python", args.script, phenotype] + args.arguments,
                stdout= subprocess.PIPE,
                stderr= subprocess.STDOUT
            )
        )
    
    for p in processes:
        p.wait()
        for line in iter(p.stdout.readline, b''):
            print (line.decode("utf-8").rstrip())
    
    print(f"Computation for all phenotypes finished in {print_elapsed_time(start_time)}")