import torch
from .CUDAcpl import tensordot,e_i_theta,_U_

def Rx(theta):
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.5,0],[-0.5,0]],[[-0.5,0],[0.5,0]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0.5,0]],[[0.5,0],[0.5,0]]]),0) 
    return a + b

def Ry(theta):
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0,0.5]],[[0,-0.5],[0.5,0]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[0.5,0],[0,-0.5]],[[0,0.5],[0.5,0]]]),0) 
    return a + b

def Rz(theta):
    a = tensordot(e_i_theta(theta/2),\
        _U_(torch.tensor,[[[0.,0.],[0.,0.]],[[0.,0.],[1.,0.]]]),0)
    b = tensordot(e_i_theta(-theta/2),\
        _U_(torch.tensor,[[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]]),0) 
    return a + b

CZ = _U_(torch.tensor,
    [[[1.,0],[0,0],[0,0],[0,0]],\
    [[0,0],[1.,0.],[0,0],[0,0]],\
    [[0,0],[0,0],[1.,0],[0,0]],\
    [[0,0],[0,0],[0,0],[-1.,0]]])


sigmax = _U_(torch.tensor,[[[0,0],[1.,0]],[[1.,0],[0,0]]])
sigmay = _U_(torch.tensor,[[[0,0],[0,-1.]],[[0,1.],[0,0]]])
sigmaz = _U_(torch.tensor,[[[1.,0],[0,0]],[[0,0],[-1.,0]]])



