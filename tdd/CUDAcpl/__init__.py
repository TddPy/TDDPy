from . import main 
import torch
from .config import device

print()
print('# DEVICE : ' + device)
print()

torch.set_printoptions(precision=15)


_U_ = main ._U_
np2CUDAcpl = main .np2CUDAcpl
CUDAcpl2np = main .CUDAcpl2np
norm = main .norm
einsum1 = main .einsum1
einsum = main .einsum
einsum3 = main .einsum3
tensordot = main .tensordot
scale = main .scale
e_i_theta = main .e_i_theta
eye = main .eye
conj = main .conj