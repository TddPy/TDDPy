from operator import index
from typing import Any,Tuple, List, Union
from .tdd import TDD
from .CUDAcpl import np2CUDAcpl,CUDAcpl_Tensor
import torch
import numpy as np

Tensor = torch.Tensor

def as_tensor(data : Union[CUDAcpl_Tensor,np.ndarray,Tuple]) -> TDD:
    '''
    construct the tdd tensor

    tensor:
        1. in the form of a matrix only: assume the parallel index and index order to be []
        2. in the form of a tuple (data, index_shape, index_order)
        Note that if the input matrix is a torch tensor, 
                then it must be already in CUDAcpl_Tensor(CUDA complex) form.
    '''

    
    return TDD.as_tensor(data)
    
    