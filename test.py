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
U2 = np.kron(U,np.kron(U,U)).reshape((2,2,2,2,2,2))
s = np.array([2,3])
#B = tdd.CUDAcpl.einsum3('...ab,...cd,...ef->...acebdf',U,U,U)
#B = tdd.CUDAcpl.einsum('...ab,...cd->...acbd',U,U)

#print(B)
print(np.tensordot(U,s,0))
print('============')
tdd1=tdd.as_tensor((U2,[],[0,3,1,4,2,5]))

tdd1.show()
print(tdd1.numpy())
exit()

tdd2=tdd.as_tensor((s,[2],[]))

tdd_final = tdd.direct_product(tdd1,tdd1,parallel_tensor=True)

tdd_final.show(path='output_test',full_output=True)

print(tdd_final.numpy())
print('yes')

exit()
