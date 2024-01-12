import argparse
from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (5e-3, 'learning rate'),
        'dropout': (0.1, 'dropout probability'),
        'cuda': (1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (300, 'maximum number of epochs to train for'),
        'optimizer': ('adam', 'which optimizer to use, can be any of [adam, radam]'),
        'batch-size': (-1, 'batch size (-1 for None)'),
        'weight-decay': (1e-3, 'l2 regularization strength')
    },
    'model_config': {
        'r': (0., 'fermi-dirac decoder parameter'),
        't': (1., 'fermi-dirac decoder parameter'),
        'layer_dims': ([[None,32,32],[32,32,None],[32,32]], 'first element of list specifies the dimensions of layers for the encoder, second specifies that for contextual decoder, third specifies that for structural decoder; the model will replaces two None automatically by the number of features'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use; tanh, relu, leaky_relu, (or None for no activation)'),
        'net-name': ('GNN', 'Specify network architecture, can be any of ["GNN", "GNN_mp"]; "GNN" for model without message passing and "GNN_mp" for model with message passing'),
        'manifold': ('Euclidean', 'Specify the type of the linear layer, can be any of ["Euclidean", "Lorentz", "PoincareBall"]'),
        'c': (1, 'hyperbolic radius for PoincareBall model'),
        'alpha': (0, 'balancing parameter in the loss and outlier score function, can be any number between 0 and 1'),
    },
    'data_config':{
        'dataset_name': ('cora', 'Specify which dataset to use, can be any of ["squirrel", "chameleon", "actor", "cora", "citeseer", "pubmed", "amazon"]'),
        'data_path': ('data/', 'Specify dataset path'),
        'featurenorm': (True, 'Specify apply l2 normalization to the features before training (True) or not (False)'),
        'outlier_type': ('contextual', 'Specify which type of outlier node to generate, can be any of ["contextual", "structural", "cont_struc", "path", "dice", "path_dice"]; "cont_struc" refers to generating both contextual and structural outliers and "path_dice" refers to generating both "path" and DICE-n outliers'),
        'outlier_preset': (True, 'whether to use preset parameters for outlier generation (True) or not (False); if True, the parameters specified below will be ignored; if False, can specify parameters below'),
        'outlier_ratio': (0.05, 'percentage of outlier node number against total node number'),
        'struc_clique_size': (20, 'parameter for structural outlier'),
        'struc_drop_prob': (0.2, 'parameter for structural outlier'),
        'sample_size': (10, 'parameter for contextual and "path" outliers'),
        'dice_ratio': (0.5, 'parameter for DICE-n outlier'),
        'outlier_seed': (None, 'random seed')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)