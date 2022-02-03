
#from tdd import index,node,tdd,methods

from . import CUDAcpl
from .tdd import TDD
from . import methods

#the data type we are using
CUDAcpl_Tensor = CUDAcpl.CUDAcpl_Tensor

# the interfaces
## system control
reset = methods.reset

## tensor manipulation
as_tensor = methods.as_tensor
direct_product = methods.direct_product
## TDD.index
sum = methods.sum
tensordot = methods.tensordot

