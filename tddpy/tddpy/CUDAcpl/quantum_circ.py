from __future__ import annotations
from typing import Any, NewType, Sequence, Union, Tuple, List
import torch
import numpy as np
from .main import tensordot,e_i_theta,_U_,CplTensor

def Rx(theta: torch.Tensor|np.ndarray) -> CplTensor:
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.5,0],[-0.5,0]],[[-0.5,0],[0.5,0]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0.5,0]],[[0.5,0],[0.5,0]]]),0) 
    return a + b

def Ry(theta: torch.Tensor|np.ndarray) -> CplTensor:
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0,0.5]],[[0,-0.5],[0.5,0]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0,-0.5]],[[0,0.5],[0.5,0]]]),0) 
    return a + b

def Rz(theta: torch.Tensor|np.ndarray) -> CplTensor:
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.,0.],[0.,0.]],[[0.,0.],[1.,0.]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]]),0) 
    return a + b

def CZ() -> CplTensor():
    return _U_(torch.tensor,
    [[[1.,0],[0,0],[0,0],[0,0]],\
    [[0,0],[1.,0.],[0,0],[0,0]],\
    [[0,0],[0,0],[1.,0],[0,0]],\
    [[0,0],[0,0],[0,0],[-1.,0]]])


def sigmax() -> CplTensor:
    return _U_(torch.tensor,[[[0,0],[1.,0]],[[1.,0],[0,0]]])

def sigmay() -> CplTensor:
    return _U_(torch.tensor,[[[0,0],[0,-1.]],[[0,1.],[0,0]]])

def sigmaz() -> CplTensor:
    return _U_(torch.tensor,[[[1.,0],[0,0]],[[0,0],[-1.,0]]])

def hadamard() -> CplTensor:
    return _U_(torch.tensor, [[[1.,0.],[1.,0.]],[[1.,0.],[-1.,0.]]])/2**0.5



