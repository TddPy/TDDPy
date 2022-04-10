from __future__ import annotations
from typing import Any, NewType, Sequence, Union, Tuple, List
import numpy as np
import torch

from .config import Config

# indicate the complex tensors in the torch.Tensor form
CplTensor = torch.Tensor

def _U_(p,*a):
    '''
    to unify the torch calculation with specified device and dtype
    p : torch method
    *a : parameters for p
    '''
    return p(*a, device = Config.device, dtype = Config.dtype)

def np2CUDAcpl(array: np.ndarray) -> CplTensor:
    '''
    transform the input array (numpy array) into the CUDA complex tensor (the form tailored for CUDA calculation)
    note: Here the complex item is stored with the extra dimension in the end.
    '''
    r = _U_(torch.tensor,np.real(array))
    i = _U_(torch.tensor,np.imag(array))
    return torch.stack((r,i),dim=-1)

def CUDAcpl2np(tensor: CplTensor) -> np.ndarray:
    '''
    transform the input array (CUDA complex tensor) into the numpy array form
    '''
    return tensor[...,0].cpu().numpy() + 1j*tensor[...,1].cpu().numpy()

def norm(tensor: CplTensor) -> torch.Tensor:
    '''
    calculate the norm, return the tensor (not in CUDA cpl form)
    '''
    return torch.sqrt(tensor[...,0]**2+tensor[...,1]**2)

def einsum1(equation: str, a: CplTensor) -> CplTensor:
    real = torch.einsum(equation,a[...,0])
    imag = torch.einsum(equation,a[...,1])
    return torch.stack((real,imag),dim=-1)

def einsum(equation: str, a: CplTensor, b: CplTensor) -> CplTensor:
    '''
    einsum for CUDA complex tensors
    '''
    real = torch.einsum(equation, a[..., 0], b[..., 0])\
            - torch.einsum(equation, a[..., 1], b[..., 1])
    imag = torch.einsum(equation, a[..., 0], b[..., 1])\
            + torch.einsum(equation, a[..., 1], b[..., 0])
    return torch.stack((real, imag), dim=-1)

def einsum3(equation: str, a: CplTensor, b: CplTensor, c: CplTensor) ->CplTensor:
    '''
    einsum for CUDA complex tensors
    '''
    real = torch.einsum(equation, a[..., 0], b[..., 0],c[..., 0])\
            - torch.einsum(equation, a[..., 1], b[..., 1],c[..., 0])\
            - torch.einsum(equation, a[...,0], b[...,1], c[...,1])\
            - torch.einsum(equation, a[...,1], b[...,0], c[...,1])
    imag = torch.einsum(equation, a[..., 0], b[..., 1], c[...,0])\
            + torch.einsum(equation, a[..., 1], b[..., 0], c[...,0])\
            + torch.einsum(equation, a[..., 0], b[..., 0], c[...,1])\
            - torch.einsum(equation, a[..., 1], b[..., 1], c[...,1])
    return torch.stack((real, imag), dim=-1)

def einsum_sublist(a: CplTensor, sublist_a: Sequence[int],
                    b : CplTensor, sublist_b: Sequence[int], output_list: Sequence[int]) -> CplTensor:
    '''
    einsum for CUDA complex tensors, in the sublist form.
    '''
    real = torch.einsum(a[..., 0], sublist_a, b[..., 0], sublist_b, output_list)\
            - torch.einsum(a[..., 1], sublist_a, b[..., 1], sublist_b, output_list)
    imag = torch.einsum(a[..., 0], sublist_a, b[..., 1], sublist_b, output_list)\
            + torch.einsum(a[..., 1], sublist_a, b[..., 0], sublist_b, output_list)
    return torch.stack((real, imag), dim=-1)

def mul_element_wise(a: CplTensor, b: CplTensor) -> CplTensor:
    '''
        return a * b (element wise)
    '''
    real = a[...,0]*b[...,0] - a[...,1]*b[...,1]
    imag = a[...,0]*b[...,1] + a[...,1]*b[...,0]
    return torch.stack((real, imag), dim=-1)

def div_element_wise(a: CplTensor, b: CplTensor) -> CplTensor:
    '''
        reutrn a / b (element wise)
    '''
    denominator = b[...,0]**2 + b[...,1]**2
    real = (a[...,0]*b[...,0] + a[...,1]*b[...,1])/denominator
    imag = (a[...,1]*b[...,0] - a[...,0]*b[...,1])/denominator
    return torch.stack((real,imag),dim=-1)

def tensordot(a: CplTensor,b: CplTensor, dim: Any =2) -> CplTensor:
    real = torch.tensordot(a[...,0],b[...,0],dim)-torch.tensordot(a[...,1],b[...,1],dim)
    imag = torch.tensordot(a[...,0],b[...,1],dim)+torch.tensordot(a[...,1],b[...,0],dim)
    return torch.stack((real, imag), dim=-1)    

def scale(s: int|float, tensor: CplTensor) -> CplTensor:
    real = tensor[...,0]*s.real-tensor[...,1]*s.imag
    imag = tensor[...,0]*s.imag+tensor[...,1]*s.real
    return torch.stack((real, imag), dim=-1)    

def e_i_theta(theta: torch.Tensor|np.ndarray) -> CplTensor:
    '''
    theta: common tensor
    output: CUDA complex tensor
    '''
    if not isinstance(theta, torch.Tensor):
        theta = _U_(torch.tensor, theta)
    return torch.stack((torch.cos(theta),torch.sin(theta)),-1)

def eye(n: int) -> CplTensor:
    result = _U_(torch.eye,n)
    return torch.stack((result,torch.zeros_like(result)),dim=-1)
    
def ones(shape: Sequence[int]) -> CplTensor:
    return torch.stack((_U_(torch.ones,shape),_U_(torch.zeros,shape)),-1)

def zeros(shape: Sequence[int]) -> CplTensor:
    return _U_(torch.zeros, tuple(shape)+(2,))

def conj(tensor: CplTensor) -> CplTensor:
    return torch.stack((tensor[...,0],-tensor[...,1]),dim=-1)








def extension(qubit_op, total, index):
    '''
    used to extend the single qubit operator qubit_op to the operator with 'total' qubits, and placed at 'index. 
    '''
    return np.kron(np.kron(np.eye(2**(total-index-1)),qubit_op),np.eye(2**index))

def special_vec_rep(vec):
    '''
    return the special vector representation, with the same global phase gauge.
    '''
    return vec/(vec[0]/np.sqrt(vec[0]*np.conj(vec[0])))