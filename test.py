import numpy as np
#from tdd import Ini_TDD,Index,Tensor
import tdd
from tdd.CUDAcpl import np2CUDAcpl
import torch

U=np2CUDAcpl(1/np.sqrt(2)*np.array([[1,1],[-1,1]]))
tdd1=tdd.as_tensor((U,[2,2],[]))
tdd1.show()

print('yes')

exit()
