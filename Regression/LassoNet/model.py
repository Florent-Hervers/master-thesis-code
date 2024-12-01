from lassonet.lassonet import LassoNetRegressor
import torch
from functools import partial
from math import isnan


def collate_params(bo_config, **params):
    """
    Hyper-parameters given by the Bayesian Optimizer should be collated according to parameter types(e.g. discrete/continous)
    Return the collated params dict.
    """
    bo_param_scope = bo_config['param_scope']
    collated_params = {}

    for k, v in params.items():
        if bo_param_scope[k]['is_discrete']:
            v = round(v)
        collated_params[k] = v

    return collated_params


def cv_eval_model(ds_tuple, bo_config, **params):
    train_feature, train_label, val_feature, val_label = ds_tuple

    params = collate_params(bo_config, **params)

    model = LassoNetRegressor(
        hidden_dims=(1024, 1024, 1024, 1024, 768 ,512, 512, 512, 512),
        lambda_start= 1.0,
        path_multiplier= 1.2,
        M=params['M'],
        optim=partial(torch.optim.Adam, lr=params['learning_rate']),
        batch_size= 64,
        dropout=0.25,
        device='cuda',
        random_state=42,
        torch_seed=42,
    )

    results = model.path(train_feature, train_label, X_val=val_feature, y_val=val_label)
    
    results_to_print = sorted(results, key=lambda a: a.correlation)
    
    # Case where all features are not chosen and the model only return a constant vector
    if isnan(results_to_print[-1].correlation):
        results_to_print.pop() 

    print("//////////////////////////////////////////////////////////////////////////////////")
    for item in results_to_print[:-1]:
        print(f"Lambda: {item.lambda_:.4f} / Correlation: {item.correlation:.3f} / Nb features: {sum(item.selected)}")
    print("Best model correlation", results_to_print[-1].correlation)
    print("Lambda =", results_to_print[-1].lambda_)
    print(f"Nb features: {sum(results_to_print[-1].selected)}")
    print("//////////////////////////////////////////////////////////////////////////////////")
    
    return results_to_print[-1].correlation