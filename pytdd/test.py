import numpy as np
import os
import torch

from pytdd import interface


from ctdd import test,as_tensor


a = torch.rand((2,3,2))
print(a)
print('start')
z = as_tensor(a,0,[0,1,2])
print('\n')
print(z)



os.system('pause')