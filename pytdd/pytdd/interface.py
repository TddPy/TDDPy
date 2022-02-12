from __future__ import annotations
from typing import Any, Dict, Tuple, List, Union, Sequence;
import numpy as np
from . import CUDAcpl;
from .CUDAcpl import CUDAcpl_Tensor

# the C++ package
from . import ctdd

class TDD:
    def __init__(self, _pointer):
        self.pointer : int = _pointer;

def as_tensor(data : TDD|CUDAcpl_Tensor|np.ndarray|
    Tuple[CUDAcpl_Tensor|np.ndarray, int, Sequence[int]]) -> TDD:
    '''
    construct the tdd tensor

    tensor:
        0. in the form of a TDD, then return a copy of it.
        1. in the form of a matrix only: assume the parallel_index_num to be 0, and index order to be []
        2. in the form of a tuple (data, index_shape, index_order)
        Note that if the input matrix is a torch tensor, 
                then it must be already in CUDAcpl_Tensor(CUDA complex) form.

    '''

    # pre-process
    if isinstance(data,TDD):
        raise Exception('not implemented')

    if isinstance(data,Tuple):
        tensor,parallel_i_num,index_order = data
    else:
        tensor = data
        parallel_i_num = 0
        index_order = []

    index_order = list(index_order);
            
    if isinstance(tensor,np.ndarray):
        tensor = CUDAcpl.np2CUDAcpl(tensor)


    # examination

    data_shape = list(tensor.shape[parallel_i_num:-1])

    if len(data_shape)!=len(index_order) and len(index_order)!=0:
        raise Exception('The number of indices must match that provided by tensor.')

    pointer = ctdd.as_tensor(tensor, parallel_i_num, index_order);

    return TDD(pointer);

def to_CUDAcpl(tensor: TDD)->CUDAcpl_Tensor:
    '''
        Transform this tensor to a CUDA complex and return.
    '''
    return ctdd.to_CUDAcpl(tensor.pointer);

def tensordot(a: TDD, b: TDD,
                axes: int|Sequence[Sequence[int]]) -> TDD:
    if isinstance(axes, int):
        pointer = ctdd.tensordot_num(a.pointer, b.pointer, axes)
    else:
        i1 = list(axes[0]);
        i2 = list(axes[1]);
        if len(i1) != len(i2):
            raise Exception("The list of indices provided")
        pointer = ctdd.tensordot_ls(a.pointer, b.pointer, i1, i2)

    return TDD(pointer)