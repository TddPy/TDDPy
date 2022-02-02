
#from tdd import index,node,tdd,methods

from tdd.CUDAcpl.main import tensordot
from . import CUDAcpl
from .node import Node,TERMINAL_ID
from .tdd import TDD
from . import methods

#the data type we are using
CUDAcpl_Tensor = CUDAcpl.CUDAcpl_Tensor

#the interfaces

as_tensor = methods.as_tensor
direct_product = methods.direct_product
#TDD.index
sum = methods.sum
tensordot = methods.tensordot

Node.reset()

