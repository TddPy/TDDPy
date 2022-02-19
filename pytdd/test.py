import numpy as np
import os
import torch

from pytdd import interface
from pytdd import CUDAcpl


a = torch.rand((2,3,2))
b = torch.rand((3,3,2))
print(CUDAcpl.tensordot(a,b.permute([1,0,2]),[[1],[1]]))

print('start')
x = interface.as_tensor((a,0,[1,0]))
y = interface.as_tensor(b)
y_perm = interface.permute(y, [1,0])
z = interface.tensordot(x,y_perm,[[1],[1]])
print(interface.to_CUDAcpl(z))

d = interface.get_tdd_info(z)
print(d)
print("\n")
print(interface.get_node_info(z.node))
z.show()



os.system('pause')