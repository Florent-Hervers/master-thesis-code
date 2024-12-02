# Master thesis code repository
This repository contain all files used for my master thesis.

## Environement setup
To setup an environement to run the files of this repo. You can simply run the instructions in `Scripts/setup_main_config.sh` if you use an conda environement. There is no need to install cupy if your computer doesn't have access to a cuda gpu. To install the dependency of all files of the repo, just run `pip install -r requirements.txt`.

## Repository structure

### Folders:
 - **Analysis notebooks:** Contain notebooks used to visualize the results obtained by the differents models. Note that I use [Weigth & biases](https://wandb.ai/flo230702/TFE/overview) to collect and visualise some models results.
 - **deepGBULP:** Contain the source code used for the paper [deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction`](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-023-00825-y), modified to fit the project. The original source code can be found [here](https://github.com/gywns6287/deepGBLUP).
 - **GBLUP**: Folder containing all output of the GBLUP computation. The software used is [ldak6](https://dougspeed.com/downloads2/). The script used to perform the computations can be found in `Scripts/GBLUP.sh`. (Note that only the relevant files used in the analysis are stored here as some generated files are heavy and not used).
 - **Regression**: Contain the source code for the LassoNet used in the paper [Tabular deep learning: a comparative study
applied to multi-task genome-wide prediction](https://pubmed.ncbi.nlm.nih.gov/39367318/), modified to fit the project. The original source code can be found [here](https://github.com/angelYHF/Tabular-deep-learning-for-GWP). The LassoNet implementation used for this paper is the one used in the paper [introducing the LassoNet architecture](https://arxiv.org/abs/1907.12207). The original repository of this code can be found [here](https://github.com/lasso-net/lassonet/blob/master/lassonet/interfaces.py)
 - **Results:**: Contain all results files used in the notebooks of the `Analysis notebooks` folder.
 - **Scripts:** Contain various scripts used thoughout the project. Note that there are intended to be launched from the root folder of the repo and are only stored in a separated folder to keep a nice structure.

 ### Files:
- **dataset.py**: implements a class representing the dataset that is usable by ML models and by pytorch deep learning models.
- **utils.py**: contain various useful functions reused in several files accros the repository.
- **requirements.txt**: described all dependencies to run every code of the repository.
- Notebooks endings by `local.ipynb` are desinged to be used to test models locally (without any gpu available).
- Pythons files are implementing the model who run on the cluster. All results are generated from the models described in those files.
- `.sbatch` files described the resources asked to the Slurm job to run the models in the cluster. Note that some of the requirements may be way higher than required if the model required low computing power/memory.

### Citation/acknoledgements
- The ResGS model was originally proposed in this [paper](https://link.springer.com/article/10.1007/s00122-024-04649-2). The code used here is a translation in pytorch of their tensorflow implementation that's can be modified. All modifications done can be followed thanks to the commit history of this repository. The original code repository of the model can be found [here](https://github.com/996184745/code-for-ResGS)