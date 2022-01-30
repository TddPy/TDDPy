import numpy as np
#from tdd import Ini_TDD,Index,Tensor
import tdd
from tdd.CUDAcpl import np2CUDAcpl
import torch

U=np2CUDAcpl(1/np.sqrt(2)*np.array([[[1,1],[-1,1]],[[1,1],[1,-1]]]))
#B = tdd.CUDAcpl.einsum3('...ab,...cd,...ef->...acebdf',U,U,U)
B = tdd.CUDAcpl.einsum('...ab,...cd->...acbd',U,U)
tdd1=tdd.as_tensor((B[0],[],[0,2,1,3]))
tdd1.show(path='output_01',full_output=True)

print('yes')

exit()
