import importlib
import numpy as np
import time
import random
from qiskit import QuantumCircuit
import torch
from qiskit.circuit.random import random_circuit
from qiskit import Aer, transpile
import qiskit.quantum_info as qi
from qiskit.circuit.random import random_circuit
from qiskit import Aer, transpile
import qiskit.quantum_info as qi
from os import system
import math


import tddpy
from tddpy import CUDAcpl
#from CUDAcpl import CUDAcpl2np

import tqdm


def random_quantum_u(width: int, depth: int) -> np.ndarray:
    circ = random_circuit(width, depth, max_operands=3, measure=False)
    op = qi.Operator(circ)
    axes = []
    for i in range(width):
        axes.append(i)
        axes.append(i+width)
    return op.data.reshape((2,)*width*2).transpose(axes)

#timing method
def timing(method, count=1):
    t1 = time.perf_counter()
    for i in range(count):
        method()
    t2 = time.perf_counter()
    print('total time: {}s, average time: {}s'.format(t2-t1, (t2-t1)/count))
    return (t2-t1)/count

def PyTorch():
    global result, gates_1, gates_2
    indices1 =[]
    indices2 =[]
    for i in range(width):
        indices1.append(1+i*2+1)
        indices2.append(1+i*2)

    result = CUDAcpl.tensordot(gates_1, gates_2, [indices1, indices2])
    axes = [0,width+1]+list(range(1,width+1))+list(range(width+2,2*width+2))+[len(result.shape)-1]
    result = result.permute(tuple(axes))
    result = CUDAcpl.CUDAcpl2np(CUDAcpl.einsum1("ii...->i...",result))


def tddpy_construct():
    global tdd1,tdd2, gates_1, gates_2
    tdd1 = tddpy.TDD.as_tensor((gates_1[0],0,[]))
    tdd2 = tddpy.TDD.as_tensor((gates_2[0],0,[]))

def tddpy_contract():
    global result
    indices1 =[]
    indices2 =[]

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = tddpy.TDD.tensordot(tdd1, tdd2, [indices1, indices2])

def tddpy_construct_T():
    global tdd1,tdd2, gates_1, gates_2
    tdd1 = tddpy.TDD.as_tensor((gates_1.to(dtype=torch.float64),1,[]))
    tdd2 = tddpy.TDD.as_tensor((gates_2.to(dtype=torch.float64),1,[]))

def tddpy_contract_T():
    global result
    indices1 =[]
    indices2 =[]

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = tddpy.TDD.tensordot(tdd1, tdd2, [indices1, indices2])

def tddpy_construct_CUDA():
    global tdd1,tdd2, gates_1, gates_2
    tdd1 = tddpy.TDD.as_tensor((gates_1.to(dtype=torch.float64).cuda(),1,[]))
    tdd2 = tddpy.TDD.as_tensor((gates_2.to(dtype=torch.float64).cuda(),1,[]))

def tddpy_contract_CUDA():
    global result
    indices1 =[]
    indices2 =[]

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = tddpy.TDD.tensordot(tdd1, tdd2, [indices1, indices2])




######## main


if __name__ == "__main__":

    count = 100
    depth = 3

    do_pytorch = False
    tddpy.setting_update(4, True, vmem_limit_MB=90000)

    width = 4
    gates_1_np_core = np.load('gate_1_np_core.npy')
    gates_2_np_core = np.load('gate_2_np_core.npy')

    with open("D:/r_d{}_count_CUDA.csv".format(depth), "a") as pfile:
        pfile.write("count, width, torch_time, time_tddpy_construct_CUDA, time_tddpy_contract_CUDA\n")
        while (True):
            #system("pause")
            gates_1_np = np.expand_dims(gates_1_np_core,0).repeat(count,0)
            gates_2_np = np.expand_dims(gates_2_np_core,0).repeat(count,0)
            gates_1 = CUDAcpl.np2CUDAcpl(gates_1_np)
            gates_2 = CUDAcpl.np2CUDAcpl(gates_2_np)


            #===================================
            print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
            print('width: {}, depth: {}, count: {}'.format(width, depth, count))
            print('input tensor shape: ', gates_1_np.shape)
            print()


            if do_pytorch:
                print('===================================================')
                print('PyTorch:')
                time_pytorch = timing(PyTorch,1)
                result_pytorch = result

                tddpy.clear_cache()
            else:
                time_pytorch = 0            

            
            tddpy.clear_cache()
            print('===================================================')
            print('tddpy tensor weight CUDA, construction:')
            time_tddpy_construct_CUDA = timing(tddpy_construct_CUDA, 1)
            #tdd1.show()

            print('tddpy tensor weight CUDA, contraction:')
            time_tddpy_contract_CUDA = timing(tddpy_contract_CUDA, 1)
            print("result tdd size: {}".format(result.size()))
            

            #print("<<diff tddpy>> ")
            #print(np.max(result_pytorch - result.numpy()))

            pfile.write("{}, {}, {}, {}, {}\n"
                        .format(count, width, time_pytorch,
                                time_tddpy_construct_CUDA, time_tddpy_contract_CUDA,
                                ))
                
            pfile.flush()

            count = int(math.ceil(count*1.5))

