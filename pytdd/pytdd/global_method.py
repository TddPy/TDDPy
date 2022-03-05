from . import ctdd

def reset(device_cuda: bool = False, eps = 3E-7):
    if device_cuda:
        ctdd.reset(1, eps)
    else:
        ctdd.reset(0, eps)

