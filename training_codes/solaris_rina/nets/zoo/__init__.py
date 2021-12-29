import os

from .hrnet import HighResolutionNet

model_dict = {
    'hrnet': {
        'weight_path': None,
        'weight_url': None,
        'arch': HighResolutionNet
    }
}
