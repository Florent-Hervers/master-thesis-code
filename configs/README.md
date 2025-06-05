# Configs folder
This folder contains the yaml configuration files used in this thesis. The library [hydra](https://hydra.cc/) was used to manage the configurations, instantiate the Python class of the models and the dataset based on the configuration, and call the training function with the appropriate parameters.

## Folder structure
The configurations are split into four folders:

- **data/**: contains the class and the parameters to instantiate for the training and validation dataset.
- **model_config/**: contains the class and the parameters of the models to instantiate. Note that the hyperparameters tuned during a sweep must be on the highest level such that they can be modified by weight and biases. The configuration evaluated on the test set starts with `Final`.
- **sweeps/**: configuration for the sweep of weight and biases. Note that the name of the hyperparameters must match the ones in the configuration file of `model_config` to make sure that weight and biases are able to change their values with the selected ones.
- **train_function_config/**: define the training function to be used and its arguments. A double question mark means that the arguments must be provided at run time. 

Finally, hydra defines the default configuration to structure the final configuration dictionary. In this thesis, we only have defined two default configurations:

1. default.yaml defines the default structure of a configuration for the training. A configuration from the `model_config` folder and `train_function_config` is required (otherwise an error is raised). The default configuration for the data folder is `SNP_marker`but can be overridden by providing another configuration. A configuration from the `sweep` folder may be provided if the training occurs during a sweep but isn't required. 

2. default_test.yaml defines the structure of the configuration dictionary used when testing the model. In this case, only a model configuration from the `model_config` folder is required.