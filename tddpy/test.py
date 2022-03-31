import numpy as np
import os
import torch

from tddpy import TDD
from tddpy import CUDAcpl

import tddpy

import tddpy_test

for i in range(10000):
    print(tddpy.get_config())
    print()
    print(i)
    print()
    tddpy_test.test1()
    tddpy_test.test2()
    tddpy_test.test3()
    tddpy_test.test3_q()
    tddpy_test.test4()
    tddpy_test.test5()
    tddpy_test.test6()
    tddpy_test.test7()
    tddpy_test.test8()
    tddpy_test.test9()
    tddpy_test.test10()

    tddpy_test.test1_T()
    tddpy_test.test1_T2()
    tddpy_test.test3_q_T()

    tddpy_test.test1_H()
    

    a = torch.rand(1000,2,2,2, dtype = torch.double)
    b = torch.rand(1000,2,2,2, dtype = torch.double)
    tdd_a = tddpy.TDD.as_tensor(a)
    tdd_b = tddpy.TDD.as_tensor(b)
    res = tddpy.TDD.tensordot(tdd_a, tdd_b, [[1],[1]])
    tddpy.clear_cache()

    tddpy.test()



os.system('pause')