from . import ctdd

def reset():
    ctdd.reset()

def clear_cache():
    ctdd.clear_cache()

def setting_update(thread_num = 4, device_cuda: bool = False, eps = 3E-7):
    if device_cuda:
        ctdd.setting_update(thread_num, 1, eps)
    else:
        ctdd.setting_update(thread_num, 0, eps)

