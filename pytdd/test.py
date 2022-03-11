import numpy as np
import os
import torch

from pytdd import TDD
from pytdd import CUDAcpl

import pytdd_test

#interface.reset()

'''
pytdd_test.test1()
pytdd_test.test2()
pytdd_test.test3()
pytdd_test.test3_q()
pytdd_test.test4()
pytdd_test.test5()
pytdd_test.test6()
pytdd_test.test7()
pytdd_test.test8()
'''

t = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmax, CUDAcpl.quantum_basic.sigmay,0)
a = TDD.as_tensor(((CUDAcpl.quantum_basic.sigmax,1,[]),None))
pytdd_test.compare(CUDAcpl.quantum_basic.sigmax,a.CUDAcpl());
print(a.info)
a.show("t1",full_output = True)

a = TDD.as_tensor(((CUDAcpl.quantum_basic.hadamard,1,[]),None))
pytdd_test.compare(CUDAcpl.quantum_basic.hadamard,a.CUDAcpl());
print(a.info)
a.show("t2",full_output = True)


os.system('pause')