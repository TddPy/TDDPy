from operator import index
from typing import Any,Tuple, List
from .tdd import TDD

Tensor = Any

def as_tensor(tensor : Tensor) -> TDD:
    '''
    construct the tdd tensor

    tensor:
        1. in the form of a matrix only: assume the parallel index and index order to be []
        2. in the form of a tuple (data, index_shape, index_order)
    '''
    if isinstance(tensor,Tuple):
        data,parallel_shape,index_order = tensor
    else:
        data = tensor
        parallel_shape = []
        index_order: List[int] = []
    
    return TDD.as_tensor(data,parallel_shape,index_order)
    
    