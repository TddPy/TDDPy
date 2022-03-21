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

def setting_update(thread_num:int = 4, device_cuda: bool = False, double_type: bool = True, eps = 3E-7,
                  gc_check_period = 0.5, vmem_limit_MB: int = 5000) -> None:
    device_cuda = 1 if device_cuda else 0
    double_type = 1 if double_type else 0
    ctdd.setting_update(thread_num, device_cuda, double_type, eps, gc_check_period, vmem_limit_MB)
    CUDAcpl.Config.setting_update(device_cuda, double_type)

    GlobalVar.current_config = get_config()
