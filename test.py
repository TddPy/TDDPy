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


U=1/np.sqrt(2)*np.array([[1,1],[1,-1]])
#U = np.random.rand(2,3)
U = np.kron(U,U).reshape((2,2,2,2))

print(U[0][1])
print('============')
tdd1=tdd.as_tensor((U,[],[2,0,1,3]))

tdd1.show(full_output=True)

tdd_indexed = tdd1.index( [(0,0),(1,1)] )
tdd_indexed.show(path = 'output_indexed', full_output = True)

print(tdd_indexed.numpy())
exit()