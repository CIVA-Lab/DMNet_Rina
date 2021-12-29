import numpy as np
from ._torch_losses import torch_losses
from torch import nn


def get_loss(framework, loss, loss_weights=None, custom_losses=None):
    """Load a loss function based on a config file for the specified framework.

    Arguments
    ---------
    framework : string
        Which neural network framework to use.
    loss : dict
        Dictionary of loss functions to use.  Each key is a loss function name,
        and each entry is a (possibly-empty) dictionary of hyperparameter-value
        pairs.
    loss_weights : dict, optional
        Optional dictionary of weights for loss functions.  Each key is a loss
        function name (same as in the ``loss`` argument), and the corresponding
        entry is its weight.
    custom_losses : dict, optional
        Optional dictionary of Pytorch classes or Keras functions of any
        user-defined loss functions.  Each key is a loss function name, and the
        corresponding entry is the Python object implementing that loss.
    """
    # lots of exception handling here. TODO: Refactor.
    if not isinstance(loss, dict):
        raise TypeError('The loss description is formatted improperly.'
                        ' See the docs for details.')
    if len(loss) > 1:

        # get the weights for each loss within the composite
        if loss_weights is None:
            # weight all losses equally
            weights = {k: 1 for k in loss.keys()}
        else:
            weights = loss_weights

        # check if sublosses dict and weights dict have the same keys
        if list(loss.keys()).sort() != list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')

        if framework == 'keras':
            return keras_composite_loss(loss, weights, custom_losses)
        elif framework in ['pytorch', 'torch']:
            return TorchCompositeLoss(loss, weights, custom_losses)

    else:  # parse individual loss functions
        loss_name, loss_dict = list(loss.items())[0]
        return get_single_loss(framework, loss_name, loss_dict, custom_losses)



def get_loss2(framework, loss, loss_weights=None, custom_losses=None):
    """Load a loss function based on a config file for the specified framework.

    Arguments
    ---------
    framework : string
        Which neural network framework to use.
    loss : dict
        Dictionary of loss functions to use.  Each key is a loss function name,
        and each entry is a (possibly-empty) dictionary of hyperparameter-value
        pairs.
    loss_weights : dict, optional
        Optional dictionary of weights for loss functions.  Each key is a loss
        function name (same as in the ``loss`` argument), and the corresponding
        entry is its weight.
    custom_losses : dict, optional
        Optional dictionary of Pytorch classes or Keras functions of any
        user-defined loss functions.  Each key is a loss function name, and the
        corresponding entry is the Python object implementing that loss.
    """
    # lots of exception handling here. TODO: Refactor.
    if not isinstance(loss, dict):
        raise TypeError('The loss description is formatted improperly.'
                        ' See the docs for details.')
    if len(loss) > 1:

        # get the weights for each loss within the composite
        if loss_weights is None:
            # weight all losses equally
            weights = {k: 1 for k in loss.keys()}
        else:
            weights = loss_weights

        # check if sublosses dict and weights dict have the same keys
        if list(loss.keys()).sort() != list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')


        return TorchCompositeLoss2(loss, weights, custom_losses)

    else:  # parse individual loss functions
        loss_name, loss_dict = list(loss.items())[0]
        return get_single_loss(framework, loss_name, loss_dict, custom_losses)


def get_single_loss(framework, loss_name, params_dict, custom_losses=None):

    if params_dict is None:
        if custom_losses is not None and loss_name in custom_losses:
            return custom_losses.get(loss_name)()
        else:
            return torch_losses.get(loss_name.lower())()
    else:
        if custom_losses is not None and loss_name in custom_losses:
            return custom_losses.get(loss_name)(**params_dict)
        else:
            return torch_losses.get(loss_name.lower())(**params_dict)



class TorchCompositeLoss(nn.Module):
    """Composite loss function."""

    def __init__(self, loss_dict, weight_dict=None, custom_losses=None):
        """Create a composite loss function from a set of pytorch losses."""
        super().__init__()
        self.weights = weight_dict
        print (loss_dict)
        self.losses = {loss_name: get_single_loss('pytorch',
                                                  loss_name,
                                                  loss_params,
                                                  custom_losses)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}  # values from the individual loss functions

    def forward(self, outputs, targets,mask):
        loss = 0
        for func_name, weight in self.weights.items():
            loss_now = self.losses[func_name](outputs, targets,mask)
            #print (func_name,loss_now.size())
            self.values[func_name]=loss_now
            loss += weight*self.values[func_name]

        return loss



class TorchCompositeLoss2(nn.Module):
    """Composite loss function."""

    def __init__(self, loss_dict, weight_dict=None, custom_losses=None):
        """Create a composite loss function from a set of pytorch losses."""
        super().__init__()
        self.weights = weight_dict
        print (loss_dict)
        self.losses = {loss_name: get_single_loss('pytorch',
                                                  loss_name,
                                                  loss_params,
                                                  custom_losses)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}  # values from the individual loss functions

    def forward(self, outputs, targets,mask):
        loss = 0
        for func_name, weight in self.weights.items():
            loss_now = self.losses[func_name](outputs, targets)
            #print (func_name)
            self.values[func_name]=loss_now
            loss += weight*self.values[func_name]

        return loss
