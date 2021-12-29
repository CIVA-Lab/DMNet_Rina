import torch
from hrnet import HighResolutionNet

def get_model(model_path):

    model=HighResolutionNet()

    loaded = torch.load(model_path)

    if isinstance(loaded, torch.nn.Module):  # if it's a full model already
        model.load_state_dict(loaded.state_dict())
    else:
        model.load_state_dict(loaded)
    return model

