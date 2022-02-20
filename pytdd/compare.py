import importlib
import numpy as np
import time
import random
from qiskit import QuantumCircuit
import torch

import tdd
from tdd import CUDAcpl

import tdd_origin
from tdd import CUDAcpl
from tdd_origin import TDD, TN
import tqdm

from pytdd import interface

#timing method
def timing(method, count=1):
    t1 = time.perf_counter()
    for i in range(count):
        method()
    t2 = time.perf_counter()
    print('total time: {}s, average time: {}s'.format(t2-t1, (t2-t1)/count))

def PyTorch():
    global result, gates_1, gates_2
    indices1 =[]
    indices2 =[]
    for i in range(width):
        indices1.append(1+i*2+1)
        indices2.append(1+i*2)

    result = CUDAcpl.CUDAcpl2np(CUDAcpl.tensordot(gates_1, gates_2, [indices1, indices2]))

def TDD_original_construct_contract():
    global result, gates_1_np, gates_2_np
    index_order = [str(i) for i in range(3*width)]

    TDD.Ini_TDD(index_order)

    var1 = []
    var2 = []
    for i in range(width):
        var1.append(TN.Index(str(i*3)))
        var1.append(TN.Index(str(i*3+1)))
        var2.append(TN.Index(str(i*3+1)))
        var2.append(TN.Index(str(i*3+2)))

    for i in range(count):
    
        ts1 = TN.Tensor(gates_1_np[i], var1)
        ts2 = TN.Tensor(gates_2_np[i], var2)

        tn = TN.TensorNetwork([ts1,ts2])
        result = tn.cont()


def TDD_original_construct():
    global gates_1_np, gates_2_np, tdd1
    index_order = [str(i) for i in range(3*width)]

    TDD.Ini_TDD(index_order)

    var1 = []
    var2 = []
    for i in range(width):
        var1.append(TN.Index(str(i*3)))
        var1.append(TN.Index(str(i*3+1)))
        var2.append(TN.Index(str(i*3+1)))
        var2.append(TN.Index(str(i*3+2)))

    for i in range(count):
    
        ts1 = TN.Tensor(gates_1_np[i], var1)
        tdd1 = ts1.tdd()
        ts2 = TN.Tensor(gates_2_np[i], var2)
        tdd2 = ts2.tdd()


def TDD_new_construct():
    global tdd1,tdd2, gates_1, gates_2
    tdd1 = tdd.as_tensor((gates_1,[count],[]))
    tdd2 = tdd.as_tensor((gates_2,[count],[]))

def TDD_new_contract():
    global result
    indices1 =[]
    indices2 =[]

    cache = dict()

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = tdd.tensordot(tdd1, tdd2, [indices1, indices2], cache)

def pytdd_construct():
    global tdd1,tdd2, gates_1, gates_2
    tdd1 = interface.as_tensor((gates_1[0],0,[]))
    tdd2 = interface.as_tensor((gates_2[0],0,[]))

def pytdd_contract():
    global result
    indices1 =[]
    indices2 =[]

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = interface.tensordot(tdd1, tdd2, [indices1, indices2])



######## main


#count = 1000 # <- recommended value for demonstration
count = 1
width = 2

rand_para_1 = torch.tensor(np.random.random((count, width)), device = CUDAcpl.device)
rand_para_2 = torch.tensor(np.random.random((count, width)), device = CUDAcpl.device)

gates_1 = CUDAcpl.ones(shape=(count,1,1))
gates_2 = CUDAcpl.ones(shape=(count,1,1))
for i in range(width):
    gates_1 = CUDAcpl.einsum('kab,kcd->kacbd', gates_1,
         CUDAcpl.quantum_basic.Rx(rand_para_1[:,i])).reshape((count,2**(i+1), 2**(i+1),2))
    gates_2 = CUDAcpl.einsum('kab,kcd->kacbd', gates_2,
         CUDAcpl.quantum_basic.Rx(rand_para_2[:,i])).reshape((count,2**(i+1), 2**(i+1),2))
gates_1 = gates_1.reshape((count,)+(2,)*width*2+(2,))
gates_2 = gates_2.reshape((count,)+(2,)*width*2+(2,))

gates_1_np = CUDAcpl.CUDAcpl2np(gates_1)
gates_2_np = CUDAcpl.CUDAcpl2np(gates_2)

#===================================





print('count: {}, width: {}'.format(count,width))
print('input tensor shape: ', gates_1_np.shape)
print()


print('===================================================')
print('PyTorch:')
timing(PyTorch,1)
print(str(result))
result_pytorch = result
print(result.shape)


'''
print('===================================================')
print('TDD original, construction:')
timing(TDD_original_construct, 1)
print()
#tdd1.show('original_tdd1',real_label = True)

print('TDD original, construction and contraction:')
timing(TDD_original_construct_contract, 1)
print(str(result.to_array())[:200])
print()
#result.show('original_result', real_label = True)
'''


'''
print('===================================================')
print('TDD refactorized, construction:')
timing(TDD_new_construct, 1)
print()
tdd1.show('refactorized_tdd1')

print('TDD refactorized, contraction:')
timing(TDD_new_contract, 1)
print(str(result)[:200])
print()
result.show('refactorized_result')

print('TDD size: ', tdd1.get_size())
'''


print('===================================================')
print('pytdd, construction:')
timing(pytdd_construct, 1)
print()
#tdd1.show('ctdd_tdd1')

print('pytdd, contraction:')
timing(pytdd_contract, 1)
print(str(interface.to_numpy(result)))
print()
#result.show('ctdd_result')
print(result_pytorch[0] - result.numpy())
print(np.max(result_pytorch[0] - result.numpy()))