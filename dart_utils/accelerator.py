import torch

def get_accelerator():

    assert torch.accelerator.is_available(), "No available accelerators detected."

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.mtia.is_available():
        torch.set_default_device("mtia")
    elif torch.xpu.is_available():
        torch.set_default_device("xpu")
    elif torch.mps.is_available():
        torch.set_default_device("mps")
