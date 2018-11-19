import torch

def save_model(model, path):
    if hasattr(model, 'module'):    # wrapped by nn.DataParallel
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
