from typing import List
from . import ctdd
from .tdd import TDD

def reset(tdd_ls: List[TDD] = []) -> None:
    tdd_w_ls = []
    tdd_t_ls = []
    for item in tdd_ls:
        if item.tensor_weight:
            tdd_t_ls.append(item.pointer)
        else:
            tdd_w_ls.append(item.pointer)

    ctdd.reset(tdd_w_ls, tdd_t_ls)

def clear_cache() -> None:
    ctdd.clear_cache()

def setting_update(thread_num = 4, device_cuda: bool = False, double_type: bool = True, eps = 3E-7) -> None:
    device_cuda = 1 if device_cuda else 0
    double_type = 1 if double_type else 0
    ctdd.setting_update(thread_num, device_cuda, double_type, eps)

