from . import main 
from . import CUDAcpl_Tensor
from . import quantum_circ
import torch

from .config import Config
#the data type provided
CplTensor = main.CplTensor

#the methods - CplTensor
_U_ = main ._U_
np2CUDAcpl = main .np2CUDAcpl
CUDAcpl2np = main .CUDAcpl2np
norm = main .norm
einsum1 = main .einsum1
einsum = main .einsum
einsum3 = main .einsum3
einsum_sublist = main.einsum_sublist
mul_element_wise = main.mul_element_wise
div_element_wise = main.div_element_wise

tensordot = main .tensordot
scale = main .scale
e_i_theta = main .e_i_theta

eye = main.eye
ones = main.ones
zeros = main.zeros

conj = main.conj

CUDAcplTensor = CUDAcpl_Tensor.CUDAcplTensor

# the methods - CUDAcplTensor
tensordot_para = CUDAcpl_Tensor.tensordot_para
permute_para = CUDAcpl_Tensor.permute_para
conj_para = CUDAcpl_Tensor.conj_para
