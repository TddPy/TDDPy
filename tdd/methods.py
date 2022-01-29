from typing import Any,Tuple
from .tdd import TDD

Tensor = Any

def as_tensor(tensor : Tensor) -> TDD:
    '''
    construct the tdd tensor

    tensor:
        1. in the form of a matrix only: assume the parallel index to be []
        2. in the form of a tuple (data,index_shape)
    '''
    if isinstance(tensor,Tuple):
        data,parallel_shape = tensor
    else:
        data = tensor
        parallel_shape = []
    