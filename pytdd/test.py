import numpy as np
import os
import torch

from pytdd import interface
from pytdd import CUDAcpl

from pytdd import pytdd_test

#interface.reset(True)

pytdd_test.test1()
pytdd_test.test2()
#pytdd_test.test3()



os.system('pause')