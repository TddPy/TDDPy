from __future__ import annotations
from typing import Any, NewType, Sequence, Union, Tuple, List
import numpy as np
import torch

from . import main
from .config import Config
from .main import CUDAcpl2np, np2CUDAcpl, tensordot, CplTensor, einsum_sublist

# indicate the complex tensors in the torch.Tensor form
CplTensor = torch.Tensor

class CUDAcplTensor:
    '''
    the data structure to store tensor with parallel index.
    '''
    @staticmethod
    def as_tensor(data : CUDAcplTensor|
                      CplTensor|np.ndarray|Tuple[CplTensor|np.ndarray, int, Sequence[int]]) -> CUDAcplTensor:
        # pre-process
        if isinstance(data, CUDAcplTensor):
            # note the order information is also copied
            return CUDAcplTensor(data.tensor, data.para_index_num)


        #####
        if isinstance(data,Tuple):
            # if contains the crd_info
            if len(data) == 2:
                data = data[0]

        #if contains the (tensor, parallel_index_num, storage_order) information
        if isinstance(data,Tuple):
            tensor, parallel_i_num, storage_order = data
        else:
            tensor = data
            parallel_i_num = 0
        #####

        return CUDAcplTensor(tensor, parallel_i_num)
            
    def __str__(self)-> str:
        return str((self.tensor, self.para_index_num))

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray], para_index_num: int = 0) -> CUDAcplTensor:
        if isinstance(tensor, np.ndarray):
            tensor = np2CUDAcpl(tensor)
        elif isinstance(tensor, torch.Tensor):
            pass
        else:
            raise "invalid tensor type."

        if para_index_num < 0 or para_index_num > len(tensor.shape)-1:
            raise "invalid parallel index number."

        self.tensor: CplTensor = tensor
        self.para_index_num :int = para_index_num

    @property
    def shape(self) -> tuple:
        return tuple(self.tensor.shape[self.para_index_num:-1])

    @property
    def para_shape(self) -> tuple:
        return tuple(self.tensor.shape[:self.para_index_num])

    def numpy(self) -> np.ndarray:
        '''
        transform the input array (CUDA complex tensor) into the numpy array form
        '''
        return self.tensor[...,0].cpu().numpy() + 1j*self.tensor[...,1].cpu().numpy()

def tensordot_para(a: CUDAcplTensor,b: CUDAcplTensor, dim: Any =2, parallel_tensor: bool = False) -> CUDAcplTensor:

    if isinstance(dim, int):
        dim = [[i for i in range(len(a.shape)-dim, len(a.shape))],[i for i in range(dim)]]

    # prepare the sublists
    sublist_a = [i for i in range(len(a.shape))]
    current_i = len(a.shape)
    sublist_b = [None] * len(b.shape)
    for i in range(len(dim[1])):
        sublist_b[dim[1][i]] = dim[0][i]
    for i in range(len(b.shape)):
        if sublist_b[i] == None:
            sublist_b[i] = current_i
            current_i += 1

    # prepare the result sublist
    sublist_res = []
    for i in range(len(a.shape)):
        if i not in dim[0]:
            sublist_res.append(i)
    sublist_res += list(range(len(a.shape), current_i))

    if not parallel_tensor:
        res = einsum_sublist(a.tensor, [...]+sublist_a, b.tensor, [...]+sublist_b, [...]+sublist_res)
        # take the maximum of para_index_num, because one para_index_num can possibly be 0
        return CUDAcplTensor(res, max(a.para_index_num, b.para_index_num))
    else:
        #prepare the sublists for the parallel index
        sublist_a_p = list(range(current_i, current_i+len(a.para_shape)))
        sublist_b_p = list(range(current_i+len(a.para_shape), current_i+len(a.para_shape)+len(b.para_shape)))
        res = einsum_sublist(a.tensor, sublist_a_p+sublist_a, 
                             b.tensor, sublist_b_p+sublist_b, 
                             sublist_a_p + sublist_b_p + sublist_res)
        return CUDAcplTensor(res, a.para_index_num + b.para_index_num)

def permute_para(a: CUDAcplTensor, perm: Sequence[int]) -> CUDAcplTensor:
    new_perm = [i for i in range(a.para_index_num)] + [i+a.para_index_num for i in perm] + [len(a.para_shape) + len(a.shape)]
    res = a.tensor.permute(new_perm)
    return CUDAcplTensor(res, a.para_index_num)


def conj_para(a: CUDAcplTensor) -> CUDAcplTensor:
    res = main.conj(a.tensor)
    return CUDAcplTensor(res, a.para_index_num)

