import numpy as np
#from tdd import Ini_TDD,Index,Tensor
import tdd
from tdd.CUDAcpl import np2CUDAcpl
import torch
'''
u1 = np.zeros((2,2,2))
u2 = np.zeros((2,2,2))
u1[0,0,0]=2
u2[0,1,0]=3

U_c = np.array([u1,u2])
'''

U=1/np.sqrt(2)*np.array([[1,1],[-1,1]])
U_2 = np.tensordot(U, U, 0)
print(np.tensordot(U_2, U_2, 3))
#print(U @ U @ U)

print('============')

tdd1=tdd.as_tensor((U,[],[1,0]))
tdd2 = tdd1.permute([1,0])
tdd2.show()

print(tdd2.numpy())
