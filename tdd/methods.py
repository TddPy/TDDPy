from __future__ import annotations

from typing import Any,Tuple, List, Union
from .tdd import TDD
from . import CUDAcpl, Node
from .CUDAcpl import np2CUDAcpl,CUDAcpl_Tensor
import torch
import numpy as np

Tensor = torch.Tensor

def as_tensor(data : CUDAcpl_Tensor|np.ndarray|Tuple) -> TDD:
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
    index_order = a.index_order+[i+len(a.index_order) for i in b.index_order]
    res = TDD(weights, data_shape, new_node, index_order)
    return res


def sum(a: TDD, b: TDD) -> TDD:
    '''
        Sum up tdd a and b, and return the reduced result. 
    '''
    return TDD.sum(a,b)