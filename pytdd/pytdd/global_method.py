from . import ctdd

def reset():
    ctdd.reset()

def clear_cache():
    ctdd.clear_cache()

def setting_update(thread_num = 4, device_cuda: bool = False, double_type: bool = True, eps = 3E-7):
    device_cuda = 1 if device_cuda else 0
    double_type = 1 if double_type else 0
    ctdd.setting_update(thread_num, device_cuda, double_type, eps)

