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
a = TDD.as_tensor(((CUDAcpl.quantum_basic.CZ.reshape((2,2,2,2,2)),1,[0,1,2]),None))
print(a.CUDAcpl())
print(a.info)
a.show("11",full_output = True)


os.system('pause')