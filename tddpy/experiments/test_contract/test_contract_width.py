import importlib
import numpy as np
import time
import random
from qiskit import QuantumCircuit
import torch

from qiskit.circuit.random import random_circuit
from qiskit import Aer, transpile
import qiskit.quantum_info as qi
from os import system

import tddpy
from tddpy import CUDAcpl

import tdd_origin
from tdd_origin import TDD, TN
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
        indices1.append(i*2+1)
        indices2.append(i*2)

    result = CUDAcpl.CUDAcpl2np(CUDAcpl.tensordot(gates_1, gates_2, [indices1, indices2]))

def TDD_original_construct_contract():
    global result, gates_1_np, gates_2_np, time_original_cont
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
    result, time_original_cont = tn.cont(None, True)


def tddpy_construct():
    global tdd1,tdd2, gates_1, gates_2
    tdd1 = tddpy.TDD.as_tensor((gates_1,0,[]))
    tdd2 = tddpy.TDD.as_tensor((gates_2,0,[]))

def tddpy_contract():
    global result
    indices1 =[]
    indices2 =[]

    for i in range(width):
        indices1.append(i*2+1)
        indices2.append(i*2)
    result = tddpy.TDD.tensordot(tdd1, tdd2, [indices1, indices2])




######## main


if __name__ == "__main__":

    tddpy.setting_update(4, False)

    width_max = 10
    depth = 5

    with open("D:/r_d{}.csv".format(depth), "a") as pfile:
        pfile.write("width, torch_time, size_tdd1, size_tdd2, time_ori_construct, time_ori_contract, size_res_ori, time_tddpy_construct, time_tddpy_contract, size_res_tddpy\n")
        while (True):
            for width in range(4, width_max+1):
                #system("pause")
                gates_1_np = random_quantum_u(width, depth)
                gates_2_np = random_quantum_u(width, depth)
                gates_1 = CUDAcpl.np2CUDAcpl(gates_1_np)
                gates_2 = CUDAcpl.np2CUDAcpl(gates_2_np)

                #===================================

                print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
                print('width: {}, depth: {}'.format(width, depth))
                print('input tensor shape: ', gates_1_np.shape)
                print()


                print('===================================================')
                print('PyTorch:')
                time_pytorch = timing(PyTorch,1)
                result_pytorch = result


                print('===================================================')
                print('TDD original, construction and contraction:')
                time_original_total = timing(TDD_original_construct_contract, 1)
                print(" cont time: ", time_original_cont)
                #result.show('original_result')
                size_result_original = result.size()
                print("result tdd size: {}".format(size_result_original))



                tddpy.clear_cache()

                print('===================================================')
                print('tddpy, construction:')
                time_tddpy_construct = timing(tddpy_construct, 1)
                print()
                #tdd1.show('ctdd_tdd1')

                print('tddpy, contraction:')
                time_tddpy_contract = timing(tddpy_contract, 1)
                print()
                #result.show("tddpy_result")
                size_tdd_1 = tdd1.size()
                size_tdd_2 = tdd2.size()
                print("tdd1 size: {}, tdd2 size: {}".format(size_tdd_1, size_tdd_2))
                size_result_tddpy = result.size()
                print("result tdd size: {}".format(size_result_tddpy))


                #print("<<diff tddpy>> ")
                #print(np.max(result_pytorch - result.numpy()))

                pfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n"
                            .format(width, time_pytorch, size_tdd_1, size_tdd_2, time_original_total - time_original_cont,
                                    time_original_cont, size_result_original, time_tddpy_construct,
                                    time_tddpy_contract, size_result_tddpy))
                pfile.flush()

