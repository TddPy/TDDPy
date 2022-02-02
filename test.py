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

#print(np.tensordot(U @ U, U, 0))
print(U @ U @ U)

print('============')

tdd1=tdd.as_tensor((U,[],[0,1]))

tdd_direct = tdd.direct_product(tdd1,tdd1)
tdd_direct = tdd.direct_product(tdd_direct,tdd1)


tdd_sum = tdd_direct.contract([[1,3],[2,4]])

tdd_sum.show(path = 'after', full_output = True)

print(tdd_sum.numpy())
exit()