import numpy as np
from .torch_callbacks import torch_callback_dict
import torch
torch.manual_seed(0)

def get_callbacks(framework, config):
    """Load callbacks based on a config file for a specific framework.

    Usage
    -----
    Note that this function is primarily intended for use with Keras. PyTorch
    does not use the same object-oriented training approach as Keras, and
    therefore doesn't generally have the same checkpointing objects to pass to
    model compilers - instead these are defined in model training code. See
    solaris.nets.train for examples of this. The only torch callback
    instantiated here is a learning rate scheduler.

    Arguments
    ---------
    framework : str
        Deep learning framework used for the model. Options are
        ``['keras', 'torch']`` .
    config : dict
        Configuration dict generated from the YAML config file.

    Returns
    -------
    callbacks : list
        A `list` of callbacks to pass to the compiler (Keras) or to wrap the
        optimizer (torch learning rate scheduling) for model training.
    """

    callbacks = []


    for callback, params in config['training']['callbacks'].items():
            if callback == 'lr_schedule':
                callbacks.append(get_lr_schedule(framework, config))
            else:
                callbacks.append(torch_callback_dict[callback](**params))

    return callbacks


def get_lr_schedule(framework, config):
    """Get a LR scheduling function for model training.

    Arguments
    ---------
    framework : str
        Deep learning framework used for the model. Options are
        ``['keras', 'torch']`` .
    config : dict
        Configuration dict generated from the YAML config file.

    Returns
    -------
    lr_scheduler : :class:`tensorflow.keras.callbacks.LearningRateScheduler` or
    ``torch.optim.lr_schedule`` scheduler class
        A scheduler to provide during training. For Keras, this takes the form
        of a callback passed to the optimizer; for PyTorch, it's a class object
        that wraps the optimizer. Because the torch version must wrap the
        optimizer, it's not instantiated here - the class is returned instead.

    """

    schedule_type = config['training'][
        'callbacks']['lr_schedule']['schedule_type']
    initial_lr = config['training']['lr']
    update_frequency = config['training']['callbacks']['lr_schedule'].get(
        'update_frequency', 1)
    factor = config['training']['callbacks']['lr_schedule'].get(
        'factor', 0)
    schedule_dict = config['training']['callbacks']['lr_schedule'].get(
        'schedule_dict', None)

    #framework == 'torch'
    # just get the class itself to use; don't instantiate until the
    # optimizer has been created.
    if config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'linear':
            lr_scheduler = torch.optim.lr_scheduler.StepLR
    elif config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR
    elif config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'arbitrary':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR

    return lr_scheduler



