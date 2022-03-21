import torch
from .main import tensordot,e_i_theta,_U_,CUDAcpl_Tensor

def Rx(theta: torch.Tensor) -> CUDAcpl_Tensor:
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.5,0],[-0.5,0]],[[-0.5,0],[0.5,0]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0.5,0]],[[0.5,0],[0.5,0]]]),0) 
    return a + b

def Ry(theta: torch.Tensor) -> CUDAcpl_Tensor:
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0,0.5]],[[0,-0.5],[0.5,0]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0,-0.5]],[[0,0.5],[0.5,0]]]),0) 
    return a + b

def Rz(theta: torch.Tensor) -> CUDAcpl_Tensor:
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.,0.],[0.,0.]],[[0.,0.],[1.,0.]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]]),0) 
    return a + b

def CZ() -> CUDAcpl_Tensor():
    return _U_(torch.tensor,
    [[[1.,0],[0,0],[0,0],[0,0]],\
    [[0,0],[1.,0.],[0,0],[0,0]],\
    [[0,0],[0,0],[1.,0],[0,0]],\
    [[0,0],[0,0],[0,0],[-1.,0]]])


def sigmax() -> CUDAcpl_Tensor:
    return _U_(torch.tensor,[[[0,0],[1.,0]],[[1.,0],[0,0]]])

def sigmay() -> CUDAcpl_Tensor:
    return _U_(torch.tensor,[[[0,0],[0,-1.]],[[0,1.],[0,0]]])

def sigmaz() -> CUDAcpl_Tensor:
    return _U_(torch.tensor,[[[1.,0],[0,0]],[[0,0],[-1.,0]]])

def hadamard() -> CUDAcpl_Tensor:
    return _U_(torch.tensor, [[[1.,0.],[1.,0.]],[[1.,0.],[-1.,0.]]])/2**0.5



