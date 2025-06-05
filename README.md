# Master thesis code repository
This repository contains all the files used for my master's thesis. Note that the phenotypes in the thesis were referenced as shoulder, top, buttock (rear), buttock (side), size, and musculature but in this repository, the original name, respectively ep_res, de_res, FESSEp_res, FESSEa_res, size_res, MUSC_res. 

## Environment setup
To set up an environment to run the files of this repo. You can simply run the instructions in `Scripts/setup_main_config.sh` if you use a conda environment. There is no need to install Cupy or the GPU version of Pytorch if your computer doesn't have access to a Cuda GPU. To install the dependency of all files of the repo with the versions used for this thesis, just run `pip install -r requirements.txt`.

## Repository structure

### Folders:
 - **Analysis notebooks:** Contain notebooks used to visualize the results obtained by the different models. Note that I use [Weigth & biases](https://wandb.ai/flo230702/TFE/overview) to collect and visualize the results from models trained using Pytorch.
 - **Configs**: Contain the hydra configuration files used during the experiments. More information about the structure of the folder can be found in the README found in this folder.
 - **deepGBULP:** Contain the source code used for the paper [deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction`](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-023-00825-y), modified to fit the project. The original source code can be found [here](https://github.com/gywns6287/deepGBLUP).
 - **GBLUP**: Folder containing all output of the GBLUP computation. The software used is [ldak6](https://dougspeed.com/downloads2/). The script used to perform the computations can be found in `GBLUP.sh`. (Note that only the relevant files used in the analysis are stored here as some generated files are heavy and not used).
 - **LassoNet**: Contain the source code for the LassoNet used in the paper [Tabular deep learning: a comparative study
applied to multi-task genome-wide prediction](https://pubmed.ncbi.nlm.nih.gov/39367318/), modified to fit the project. The original source code can be found [here](https://github.com/angelYHF/Tabular-deep-learning-for-GWP). The LassoNet implementation used for this paper is the one used in the paper [introducing the LassoNet architecture](https://arxiv.org/abs/1907.12207). The original repository of this code can be found [here](https://github.com/lasso-net/lassonet/blob/master/lassonet/interfaces.py)
 - **Models**: This folder contains the PyTorch implementation of the different models used in this thesis.
 - **Old files**: This folder contains files used during the thesis but that become obsolete after some refactoring of the repository/codebase.
 - **Results**: This folder contains all files that contain the results not saved in the weight and biases.
 - **Scripts:** This folder contains all Slurm scripts used to run the scripts. These files provide information about the hardware resources used for every experiment and provide examples to execute the Python scripts. The scripts containing the prefix `train_final` are the ones used to train the best configuration. 

 ### File Description:
The main files will be described here, a docstring is available for every less important script to explain the purpose of the file. A majority of the scripts require arguments when called, an explanation of all arguments can be found by executing the script with the `-h` flag.
- **dataset.py**: implements the Python classes representing the dataset that is usable by ML models and by pytorch deep learning models.
- **GPTransformer.py**: implements the different tokenization methods described in the paper and starts the training of the model based on the given configuration. 
- **sweep.py**: script that is used to launch the hyperparameters tuning thanks to the sweep option of weight and biases.
- **train.py**: main script that launches the training of a model based on the given configuration.
- **train_residual.py**: script that launches the training of the model on the residual of the ridge model based on the given configuration.
- **utils.py**: contains all functions that may be used by several scripts.
- Notebooks ending by `_local.ipynb` are designed to be used to test and debug models locally (without any GPU available).
- `random_forest.py`, `random_forest_all.py`, `Ridge.py`, `SVM.py`, `XGBoost.py`, and `XGBoost_all.py` implements the data processing, the training, and the saving of the results of their respective models.
- `_all.py` implements the same models but instead of having one model per phenotype, we use one model to predict four phenotypes directly
- Pythons files are implementing the model that runs on the cluster. All results are generated from the models described in those files.

### Citation/acknowledgments
- The ResGS model was originally proposed in this [paper](https://link.springer.com/article/10.1007/s00122-024-04649-2). The code used here is a translation in Pytorch of their TensorFlow implementation that can be modified. All modifications done can be followed thanks to the commit history of this repository. The original code repository of the model can be found [here](https://github.com/996184745/code-for-ResGS)
- The implementation of the locally connected layer is based on the implementation from the paper [deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction`](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-023-00825-y) and was extended to fit the needs of the project.