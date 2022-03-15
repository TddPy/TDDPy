import numpy as np
import os
import torch

from pytdd import TDD
from pytdd import CUDAcpl

import pytdd

import pytdd_test

#interface.reset()

for i in range(1):
    
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
    
    pytdd.setting_update(1, True, True)
    pytdd_test.test3_cuda();

    a = TDD.as_tensor(np.array([[1,2],[3,4]]))
    pytdd.reset([a])
    print(a.numpy())

os.system('pause')