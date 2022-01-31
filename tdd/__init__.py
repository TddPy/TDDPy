
#from tdd import index,node,tdd,methods

from . import CUDAcpl
from .node import Node,TERMINAL_ID
from .tdd import TDD
from . import methods

#the data type we are using
CUDAcpl_Tensor = CUDAcpl.CUDAcpl_Tensor

#the interfaces

as_tensor = methods.as_tensor
direct_product = methods.direct_product

Node.reset()

