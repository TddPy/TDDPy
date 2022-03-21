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
import tdd
from tdd import CUDAcpl

import tdd_origin
from tdd import CUDAcpl
from tdd_origin import TDD, TN
import tqdm

from tddpy import interface



def random_quantum_u(width: int, depth: int) -> np.ndarray:
    circ = random_circuit(width, depth, max_operands=3, measure=False)
    op = qi.Operator(circ)
    return op.data.reshape((2,)*width*2)

if __name__ == "__main__":
    width_max = 7
    depth = 4
    with open("D:/r_d{}.csv".format(depth),"w") as pfile:
        pfile.write("cir_width, size_trival, size_pair\n")

        for width in range(2, width_max+1):
            index_order = []
            for i in range(width):
                index_order.append(i)
                index_order.append(i+width)

            for i in range(50):
                temp1 = interface.as_tensor((random_quantum_u(width, depth), 0, []))
                temp2 = interface.as_tensor((random_quantum_u(width, depth), 0, index_order))
                size1 =  temp1.size()
                size2 =  temp2.size()
                pfile.write("{}, {}, {}\n".format(width, size1, size2))
                print("{}, {}, {}\n".format(width, size1, size2))
