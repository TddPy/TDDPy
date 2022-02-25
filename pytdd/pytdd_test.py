
import torch
from torch._C import dtype
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
    b = torch.rand((2,2,2), dtype=torch.double)
    expected = CUDAcpl.tensordot(a,b,1)
    
    tdd_a = interface.as_tensor((a,0,[1,0]))
    tdd_b = interface.as_tensor((b,0,[1,0]))
    actual = interface.tensordot(tdd_a, tdd_b, 1).CUDAcpl()

    compare(expected,actual)


def test2():
    '''
    permutation and contraction
    '''
    a = torch.rand((2,2,2,2,2))
    expected = a.permute((0,2,3,1,4))

    tdd_a = interface.as_tensor((a,0,[0,2,3,1]))
    actual = interface.permute(tdd_a, (0,2,3,1)).CUDAcpl()

    compare(expected, actual)


def test3():
    '''
    tracing, cuda
    '''
    a = torch.rand((2,2,2,2,2), dtype = torch.double, device = 'cuda')
    expected = torch.einsum("iijjk->k", a)

    tdd_a = interface.as_tensor(a)
    actual = interface.trace(tdd_a, [[0,2],[1,3]]).CUDAcpl()

    compare(expected, actual)


def test4():
    '''
    large tensor contraction
    '''

    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmay, CUDAcpl.quantum_basic.sigmax, 0)
    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, a, 0)

    b = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, CUDAcpl.quantum_basic.hadamard, 0)
    b = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, b, 0)

    expected = CUDAcpl.tensordot(a,b,[[1,5,3],[4,0,5]])
    
    tdd_a = interface.as_tensor((a,0,[]))
    tdd_b = interface.as_tensor((b,0,[]))
    actual = interface.tensordot(tdd_a, tdd_b, [[1,5,3],[4,0,5]]).CUDAcpl()

    compare(expected,actual)

def test5():

    '''
    small tensor contraction
    '''

    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmay, CUDAcpl.quantum_basic.hadamard, 0)

    b = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmax, CUDAcpl.quantum_basic.hadamard, 0)

    expected = CUDAcpl.tensordot(a,b,[[1,3],[3,2]])
    
    tdd_a = interface.as_tensor((a,0,[]))
    tdd_a.show("A")
    tdd_b = interface.as_tensor((b,0,[]))
    tdd_b.show("B")
    actual = interface.tensordot(tdd_a, tdd_b, [[1,3],[3,2]]).CUDAcpl()

    compare(expected,actual)

def test6():

    '''
    micro tensor contraction
    '''

    a = CUDAcpl._U_(torch.rand,(2,2,2))

    b = CUDAcpl.quantum_basic.hadamard

    expected = CUDAcpl.tensordot(a,b,[[0,1],[1,0]])
    
    tdd_a = interface.as_tensor((a,0,[]))
    tdd_b = interface.as_tensor((b,0,[]))
    actual = interface.tensordot(tdd_a, tdd_b, [[0,1],[1,0]]).CUDAcpl()

    compare(expected,actual)
