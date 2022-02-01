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


#U=1/np.sqrt(2)*np.array([[1,1],[1,-1]])
U1 = np.random.rand(2,3)
U2 = -U1
#U = np.kron(U,U).reshape((2,2,2,2))

print(U1+U2)
print('============')
tdd1=tdd.as_tensor((U1,[],[]))
tdd2=tdd.as_tensor((U2,[],[]))

tdd_sum = tdd.sum(tdd1,tdd2)

tdd_sum.show(full_output=True)

tdd_sum.show(path = 'output', full_output = True)

print(tdd_sum.numpy())
exit()