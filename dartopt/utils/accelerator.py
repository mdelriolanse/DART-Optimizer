import torch

def get_accelerator():
    """
    Get the best available accelerator device.
    
    Returns:
        torch.device: The best available device (cuda, mps, xpu, mtia, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device("xpu")
    elif hasattr(torch, 'mtia') and torch.mtia.is_available():
        return torch.device("mtia")
    else:
        return torch.device("cpu")