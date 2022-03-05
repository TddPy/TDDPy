import numpy as np
import os
import torch

from pytdd import TDD
from pytdd import CUDAcpl

import pytdd_test

#interface.reset(True)

#pytdd_test.test1()
#pytdd_test.test2()
#pytdd_test.test3()
#pytdd_test.test4()
#pytdd_test.test5()
#pytdd_test.test6()
#pytdd_test.test7()
#pytdd_test.test8()


tensor = CUDAcpl.einsum('ab,cd->acbd',CUDAcpl.quantum_basic.hadamard,CUDAcpl.quantum_basic.hadamard);
tdd = TDD.as_tensor(((tensor,0,[0,2,1,3]),None))
tdd.show()




os.system('pause')