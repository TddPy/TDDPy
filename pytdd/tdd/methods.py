from __future__ import annotations

from typing import Any, Dict,Tuple, List, Union, Sequence

from .node import Node
from .tdd import TDD
from . import CUDAcpl
from .CUDAcpl import np2CUDAcpl,CUDAcpl_Tensor
import torch
import numpy as np

Tensor = torch.Tensor

def reset():
    Node.reset()

def as_tensor(data : TDD|CUDAcpl_Tensor|np.ndarray|
    Tuple[CUDAcpl_Tensor|np.ndarray, Sequence[int], Sequence[int]]) -> TDD:
    '''
    construct the tdd tensor

    tensor:
        1. in the form of a matrix only: assume the parallel index and index order to be []
        2. in the form of a tuple (data, index_shape, index_order)
        Note that if the input matrix is a torch tensor, 
                then it must be already in CUDAcpl_Tensor(CUDA complex) form.
    '''

    
    return TDD.as_tensor(data)
    

def direct_product(a: TDD, b: TDD, parallel_tensor: bool = False)-> TDD:
    '''
        Return the direct product: a tensor b. The index order is the connection of that of a and b.

        parallel_tensor: whether to tensor on the parallel indices
            False: parallel index of a and b must be the same, and their shapes are:
                a: [(?), (s_a), 2] tensor b: [(?), (s_b), 2] -> [(?), (s_a), (s_b), 2]
            True: tensor on the parallel indices too. Their shapes are:
                a: [(?a), (s_a), 2] tensor b: [(?b), (s_b), 2] -> [(?a), (?b), (s_a), (s_b), 2]
    '''
    if parallel_tensor:
        weights = CUDAcpl.tensordot(a.weights,b.weights,dim=0)
    else:
        #check the equality of parallel shapes
        if a.parallel_shape != b.parallel_shape:
            raise Exception('Parallel shapes of a ' + str(a.parallel_shape)+' and b '
                            +str(b.parallel_shape)+' are not identical.')

        weights = CUDAcpl.mul_element_wise(a.weights, b.weights)
    new_node = Node.append(a.node,a.parallel_shape,len(a.data_shape),b.node,b.parallel_shape,parallel_tensor)


    data_shape = a.data_shape + b.data_shape
    index_order = a.index_order+tuple([i+len(a.index_order) for i in b.index_order])
    res = TDD(weights, data_shape, new_node, index_order)
    return res


def sum(a: TDD, b: TDD) -> TDD:
    '''
        Sum up tdd a and b, and return the reduced result.
        Note that the index_order and data_shape must be the same. 
    '''
    if a.global_shape != b.global_shape:
        raise Exception('To sum them up, a and b must be in the same shape.')
    if a.index_order != b.index_order:
        raise Exception('index_order not the same, sum in this case is not supported yet.')
    return TDD.sum(a,b)

def tensordot(a: TDD, b: TDD,
                axes: Union[int, Sequence[Sequence[int]]], 
                sum_dict_cache: Dict= None,
                parallel_tensor: bool=False) -> TDD:
    '''
        The pytorch-like tensordot method. Note that indices should be counted with data indices only.
        sum_dict_cache: the dictionary cache of former summation calculations.
        parallel_tensor: Whether to tensor on the parallel indices.
    '''
    
    indices_a: List[int] = []
    indices_b: List[int] = []

    if isinstance(axes,int):
        for i in range(axes):
            indices_a.append(a.dim_data-axes+i)
            indices_b.append(a.dim_data+i)
    else:
        for k in range(len(axes[0])):
            indices_a.append(axes[0][k])
            indices_b.append(a.dim_data + axes[1][k])
    
    temp_tensor = direct_product(a,b, parallel_tensor)

    return temp_tensor.contract([indices_a, indices_b], sum_dict_cache)