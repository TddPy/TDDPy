import numpy as np
import os
import torch

from pytdd import interface
from pytdd import CUDAcpl


a = torch.rand((2,3,2))
b = torch.rand((3,3,2))
print(CUDAcpl.tensordot(a,b,[[1],[1]]))

print('start')
x = interface.as_tensor((a,0,[1,0]))
y = interface.as_tensor(b)
z = interface.tensordot(x,y,[[1],[1]])
print(interface.to_CUDAcpl(z))



os.system('pause')