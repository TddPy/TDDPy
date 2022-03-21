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
from CUDAcpl import CUDAcpl2np

import tdd
from tdd import CUDAcpl

import tdd_origin
from tdd import CUDAcpl
from tdd_origin import TDD, TN
import tqdm

import pytdd



def random_quantum_u(width: int, depth: int) -> np.ndarray:
    circ = random_circuit(width, depth, max_operands=3, measure=False)
    op = qi.Operator(circ)
    return op.data.reshape((2,)*width*2)

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
    result = CUDAcpl2np(CUDAcpl.einsum1("ii...->i...",result))

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

    
    ts1 = TN.Tensor(gates_1_np, var1)
    ts2 = TN.Tensor(gates_2_np, var2)

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

    
    ts1 = TN.Tensor(gates_1_np, var1)
    tdd1 = ts1.tdd()
    ts2 = TN.Tensor(gates_2_np, var2)
    tdd2 = ts2.tdd()


def pytdd_construct():
    global tdd1,tdd2, gates_1, gates_2
    index_order = []
    for i in range(width):
        index_order.append(i)
        index_order.append(i+width)
    tdd1 = pytdd.TDD.as_tensor(((gates_1[0],0,[]),None))
    tdd2 = pytdd.TDD.as_tensor(((gates_2[0],0,[]),None))

def pytdd_contract():
    global result
    indices1 =[]
    indices2 =[]

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = pytdd.TDD.tensordot(tdd1, tdd2, [indices1, indices2], iteration_parallel=True)

def pytdd_construct_T():
    global tdd1,tdd2, gates_1, gates_2
    index_order = []
    for i in range(width):
        index_order.append(i)
        index_order.append(i+width)
    tdd1 = pytdd.TDD.as_tensor(((gates_1,1,[]),None))
    tdd2 = pytdd.TDD.as_tensor(((gates_2,1,[]),None))

def pytdd_contract_T():
    global result
    indices1 =[]
    indices2 =[]

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = pytdd.TDD.tensordot(tdd1, tdd2, [indices1, indices2], iteration_parallel=True)




######## main


if __name__ == "__main__":

    count = 1000
    width_max = 7
    depth = 2

    with open("D:/r_d{}_c{}.csv".format(depth, count), "a") as pfile:
        pfile.write("count, width, size_tdd1, size_tdd2, time_pytdd_construct, time_pytdd_contract, time_pytdd_construct_T, time_pytdd_contract_T, size_res_pytdd\n")
        while (True):
            for width in range(4, width_max+1):
                system("pause")
                gates_1_np = np.expand_dims(random_quantum_u(width, depth),0).repeat(count,0)
                gates_2_np = np.expand_dims(random_quantum_u(width, depth),0).repeat(count,0)
                gates_1 = CUDAcpl.np2CUDAcpl(gates_1_np)
                gates_2 = CUDAcpl.np2CUDAcpl(gates_2_np)

                #===================================

                print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
                print('width: {}, depth: {}, count: {}'.format(width, depth, count))
                print('input tensor shape: ', gates_1_np.shape)
                print()


                '''
                print('===================================================')
                print('TDD original, construction:')
                time_original_construct = timing(TDD_original_construct, 1)
                print()
                #tdd1.show('original_tdd1',real_label = True)

                print('TDD original, construction and contraction:')
                time_original_total = timing(TDD_original_construct_contract, 1)
                print()
                #result.show('original_result')
                size_result_original = result.size()
                print("result tdd size: {}".format(size_result_original))
                '''


                pytdd.reset()

                print('===================================================')
                print('pytdd scalar weight single, construction:')
                time_pytdd_construct = timing(pytdd_construct, 1)
                print()
                size_tdd_1 = tdd1.size()
                size_tdd_2 = tdd2.size()

                print('pytdd scalar weight single, contraction:')
                time_pytdd_contract = timing(pytdd_contract, 1)
                print()
                print("tdd1 size: {}, tdd2 size: {}".format(size_tdd_1, size_tdd_2))
                size_result_pytdd = result.size()
                print("result tdd size: {}".format(size_result_pytdd))


                pytdd.reset()

                print('===================================================')
                print('pytdd tensor weight, construction:')
                time_pytdd_construct_T = timing(pytdd_construct_T, 1)
                #tdd1.show()

                print('pytdd tensor weight, contraction:')
                time_pytdd_contract_T = timing(pytdd_contract_T, 1)
                print("result tdd size: {}".format(result.size()))

                pfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n"
                            .format(count, width, size_tdd_1, size_tdd_2, time_pytdd_construct,
                                    time_pytdd_contract, time_pytdd_construct_T, time_pytdd_contract_T, size_result_pytdd))
                pfile.flush()

