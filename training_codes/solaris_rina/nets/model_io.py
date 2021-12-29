import os
from tensorflow import keras
import torch
torch.manual_seed(0)
from warnings import warn
import requests
import numpy as np
from tqdm.auto import tqdm
from solaris_rina.nets.zoo import model_dict


def get_model(model_name, framework, model_path=None, pretrained=False,
              custom_model_dict=None, num_classes=1):
    """Load a model from a file based on its name."""
    if custom_model_dict is not None:
        md = custom_model_dict
    else:
        md = model_dict.get(model_name, None)
        if md is None:  # if the model's not provided by solaris
            raise ValueError(f"{model_name} can't be found in solaris and no "
                             "custom_model_dict was provided. Check your "
                             "model_name in the config file and/or provide a "
                             "custom_model_dict argument to Trainer(). ")
    if model_path is None or custom_model_dict is not None:
        model_path = md.get('weight_path')
    if num_classes == 1:
        model = md.get('arch')(pretrained=pretrained)
    else:
        model = md.get('arch')(num_classes=num_classes, pretrained=pretrained)

    if model is not None and pretrained:
        try:

            model = _load_model_weights(model, model_path, framework)
        except (OSError, FileNotFoundError):
            warn(f'The model weights file {model_path} was not found.')

    return model


def _load_model_weights(model, path, framework):
    """Backend for loading the model."""

    if torch.cuda.is_available():
        try:
            loaded = torch.load(path)
        except FileNotFoundError:

            loaded = torch.load(path)
    else:
        try:
            loaded = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            loaded = torch.load(path, map_location='cpu')

    if isinstance(loaded, torch.nn.Module):  # if it's a full model already
        model.load_state_dict(loaded.state_dict())
    else:
        model.load_state_dict(loaded)

    return model


def reset_weights(model, framework):
    """Re-initialize model weights for training.

    Arguments
    ---------
    model : :class:`tensorflow.keras.Model` or :class:`torch.nn.Module`
        A pre-trained, compiled model with weights saved.
    framework : str
        The deep learning framework used. Currently valid options are
        ``['torch', 'keras']`` .

    Returns
    -------
    reinit_model : model object
        The model with weights re-initialized. Note this model object will also
        lack an optimizer, loss function, etc., which will need to be added.
    """

    if framework == 'keras':
        model_json = model.to_json()
        reinit_model = keras.models.model_from_json(model_json)
    elif framework == 'torch':
        reinit_model = model.apply(_reset_torch_weights)

    return reinit_model


def _reset_torch_weights(torch_layer):
    if isinstance(torch_layer, torch.nn.Conv2d) or \
            isinstance(torch_layer, torch.nn.Linear):
        torch_layer.reset_parameters()
