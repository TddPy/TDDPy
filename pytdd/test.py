import numpy as np
import os
import torch

from pytdd import TDD
from pytdd import CUDAcpl

import pytdd

import pytdd_test

#interface.reset()

pytdd.setting_update(4, False, True, 3e-7, 0.5, 8000)

for i in range(10000):
    print()
    print(i)
    print()
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
    

    a = torch.rand(1000,2,2,2)
    b = torch.rand(1000,2,2,2)
    tdd_a = pytdd.TDD.as_tensor(a)
    tdd_b = pytdd.TDD.as_tensor(b)
    res = pytdd.TDD.tensordot(tdd_a, tdd_b, [[1],[1]])
    pytdd.clear_cache()

    pytdd.test()



os.system('pause')