import numpy as np
import os
import torch

from pytdd import TDD
from pytdd import CUDAcpl

import pytdd_test

#interface.reset()


pytdd_test.test1()
pytdd_test.test2()
pytdd_test.test3()
pytdd_test.test3_q()
pytdd_test.test4()
pytdd_test.test5()
pytdd_test.test6()
pytdd_test.test7()
pytdd_test.test8()
pytdd_test.test9()
pytdd_test.test10()

pytdd_test.test1_T()
pytdd_test.test1_T2()
pytdd_test.test3_q_T()

pytdd_test.test1_H()

a = torch.zeros((2,2,2,2),dtype=torch.double)
a_tdd = TDD.as_tensor(((a,0,[]),None))
print(a_tdd.CUDAcpl())

os.system('pause')