import numpy as np
import os
import torch

from pytdd import interface
from pytdd import CUDAcpl

from pytdd import pytdd_test

pytdd_test.test1()
pytdd_test.test2()

a = interface.as_tensor((CUDAcpl.quantum_basic.hadamard,0,[]))

t = interface.tensordot(a,a,0);

t_ = interface.as_tensor((t.CUDAcpl(), 0, [0,2,1,3]))



t.show()

t_.show("swapped")

os.system('pause')