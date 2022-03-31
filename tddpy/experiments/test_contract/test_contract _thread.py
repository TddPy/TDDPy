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

    count = 1000


    depth = 3
    width = 6
    gates_1_np_core = random_quantum_u(width, depth)
    gates_2_np_core = random_quantum_u(width, depth)
    #np.save('gate_1_np_core.npy', gates_1_np_core)
    #np.save('gate_2_np_core.npy', gates_2_np_core)
    #gates_1_np_core = np.load('gate_1_np_core.npy')
    #gates_2_np_core = np.load('gate_2_np_core.npy')

    max_thread_num = 10

    with open("D:/r_d{}_count.csv".format(depth), "a") as pfile:
        pfile.write("count, width, thread_num, size_tdd1, size_tdd2, time_tddpy_construct, time_tddpy_contract\n")
        for thread in range(1,max_thread_num+1):

            tddpy.clear_cache()
            tddpy.reset(thread, False, vmem_limit_MB=90000)

            #system("pause")
            #===================================
            gates_1_np = np.expand_dims(gates_1_np_core,0).repeat(count,0)
            gates_2_np = np.expand_dims(gates_2_np_core,0).repeat(count,0)
            gates_1 = CUDAcpl.np2CUDAcpl(gates_1_np)
            gates_2 = CUDAcpl.np2CUDAcpl(gates_2_np)

            print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
            print('width: {}, depth: {}, count: {}, thread: {}'.format(width, depth, count, thread))
            print('input tensor shape: ', gates_1_np.shape)
            print()

            
            print('===================================================')
            print('tddpy scalar weight single, construction:')
            time_tddpy_construct = timing(tddpy_construct, 1) * count
            print()
            size_tdd_1 = tdd1.size()
            size_tdd_2 = tdd2.size()

            print('tddpy scalar weight single, contraction:')
            time_tddpy_contract = timing(tddpy_contract, 1) * count
            print()
            print("tdd1 size: {}, tdd2 size: {}".format(size_tdd_1, size_tdd_2))
            size_result_tddpy = result.size()
            print("result tdd size: {}".format(size_result_tddpy))
            

            '''
            tddpy.clear_cache()
            print('===================================================')
            print('tddpy tensor weight, construction:')
            time_tddpy_construct_T = timing(tddpy_construct_T, 1)
            #tdd1.show()

            print('tddpy tensor weight, contraction:')
            time_tddpy_contract_T = timing(tddpy_contract_T, 1)
            print("result tdd size: {}".format(result.size()))
            '''

            '''
            tddpy.clear_cache()
            print('===================================================')
            print('tddpy tensor weight CUDA, construction:')
            time_tddpy_construct_CUDA = timing(tddpy_construct_CUDA, 1)
            #tdd1.show()

            print('tddpy tensor weight CUDA, contraction:')
            time_tddpy_contract_CUDA = timing(tddpy_contract_CUDA, 1)
            print("result tdd size: {}".format(result.size()))
            '''

            #print("<<diff tddpy>> ")
            #print(np.max(result_pytorch - result.numpy()))

            pfile.write("{}, {}, {}, {}, {}, {}, {}\n"
                        .format(count, width, thread, size_tdd_1, size_tdd_2,
                                time_tddpy_construct, time_tddpy_contract))
                
            pfile.flush()
