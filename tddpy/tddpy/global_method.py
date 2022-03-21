from typing import List
from . import ctdd
from . import CUDAcpl



def test() -> None:
    '''
        This method is for testing purpose.
    '''
    ctdd.test()

def clear_cache() -> None:
    ctdd.clear_cache()

def get_config() -> None:
    return ctdd.get_config()


# the current configuration of kernel is recorded
class GlobalVar:
    current_config = get_config()

def setting_update(thread_num:int = 4, device_cuda: bool = False, dtype_double: bool = True, eps = 3E-7,
                  gc_check_period = 0.5, vmem_limit_MB: int = 5000) -> None:
    device_cuda = 1 if device_cuda else 0
    dtype_double = 1 if dtype_double else 0
    ctdd.setting_update(thread_num, device_cuda, dtype_double, eps, gc_check_period, vmem_limit_MB)
    CUDAcpl.Config.setting_update(device_cuda, dtype_double)

    GlobalVar.current_config = get_config()
