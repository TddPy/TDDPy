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

CZ: CUDAcpl_Tensor = _U_(torch.tensor,
    [[[1.,0],[0,0],[0,0],[0,0]],\
    [[0,0],[1.,0.],[0,0],[0,0]],\
    [[0,0],[0,0],[1.,0],[0,0]],\
    [[0,0],[0,0],[0,0],[-1.,0]]])


sigmax: CUDAcpl_Tensor = _U_(torch.tensor,[[[0,0],[1.,0]],[[1.,0],[0,0]]])
sigmay: CUDAcpl_Tensor = _U_(torch.tensor,[[[0,0],[0,-1.]],[[0,1.],[0,0]]])
sigmaz: CUDAcpl_Tensor = _U_(torch.tensor,[[[1.,0],[0,0]],[[0,0],[-1.,0]]])



