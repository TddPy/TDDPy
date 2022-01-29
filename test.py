import numpy as np
#from tdd import Ini_TDD,Index,Tensor
import tdd
from tdd.CUDAcpl import np2CUDAcpl
import torch

U=np2CUDAcpl(1/np.sqrt(2)*np.array([[1,1],[-1,1]]))
B = tdd.CUDAcpl.einsum3('ab,cd,ef->acebdf',U,U,U)

tdd1=tdd.as_tensor((B,[],[0,3,1,4,2,5]))
tdd1.show()

print('yes')

exit()
