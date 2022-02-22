
import torch
from . import CUDAcpl
from . import interface

def compare(expected: CUDAcpl.CUDAcpl_Tensor,
            actual: CUDAcpl.CUDAcpl_Tensor):
    max_diff = torch.max(abs(expected - actual))

    if ( max_diff > 1e-7):
        print("not passed, diff: ",max_diff)
    else:
        print("passed, diff: ",max_diff)

def test1():
    '''
    direct contraction
    '''
    a = CUDAcpl.quantum_basic.sigmax
    b = CUDAcpl.quantum_basic.sigmay
    expected = CUDAcpl.tensordot(a,b,1)
    
    tdd_a = interface.as_tensor(a)
    tdd_b = interface.as_tensor(b)
    actual = interface.tensordot(tdd_a, tdd_b, 1).CUDAcpl()

    compare(expected,actual)


def test2():
    '''
    permutation and contraction
    '''
    a = torch.rand((2,2,2,2,2))
    expected = a.permute((0,2,3,1,4))

    tdd_a = interface.as_tensor(a)
    actual = interface.permute(tdd_a, (0,2,3,1)).CUDAcpl()

    compare(expected, actual)


def test3():
    '''
    tracing
    '''
    a = torch.rand((2,2,2,2,2), dtype = torch.double, device = 'cuda')
    expected = torch.einsum("iijjk->k", a)

    tdd_a = interface.as_tensor(a)
    actual = interface.trace(tdd_a, [[0,2],[1,3]]).CUDAcpl()

    compare(expected, actual)

