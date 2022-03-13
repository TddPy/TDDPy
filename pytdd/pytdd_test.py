
import torch
from torch._C import dtype
from pytdd import TDD, CUDAcpl

def compare(title, expected: CUDAcpl.CUDAcpl_Tensor,
            actual: CUDAcpl.CUDAcpl_Tensor):
    max_diff = torch.max(abs(expected - actual))

    if ( max_diff > 1e-7):
        print(title+" not passed, diff: ",max_diff)
    else:
        print(title+" passed, diff: ",max_diff)

def test1():
    '''
    direct contraction
    '''
    a = CUDAcpl.quantum_basic.sigmax
    b = torch.rand((2,2,2), dtype=torch.double)
    expected = CUDAcpl.tensordot(a,b,1)
    
    tdd_a = TDD.as_tensor(((a,0,[1,0]),None))
    tdd_b = TDD.as_tensor(((b,0,[1,0]),None))
    actual = TDD.tensordot(tdd_a, tdd_b, 1).CUDAcpl()

    compare("test1", expected, actual)

def test1_T():
    '''
    direct contraction
    '''
    a = CUDAcpl.quantum_basic.sigmax
    b = torch.rand((2,2,2), dtype=torch.double)
    expected = CUDAcpl.einsum("ia,ia->i",a,b)

    tdd_a = TDD.as_tensor(((a,1,[]),None))
    tdd_b = TDD.as_tensor(((b,1,[]),None))
    actual = TDD.tensordot(tdd_a, tdd_b, [[0],[0]],[], False).CUDAcpl()

    compare("test1_T",expected,actual)

def test1_T2():
    '''
    direct contraction
    '''
    a = CUDAcpl.quantum_basic.sigmax
    b = torch.rand((2,2,2), dtype=torch.double)
    expected = CUDAcpl.einsum("ia,ja->ij",a,b)

    tdd_a = TDD.as_tensor(((a,1,[]),None))
    tdd_b = TDD.as_tensor(((b,1,[]),None))
    actual = TDD.tensordot(tdd_a, tdd_b, [[0],[0]],[], True).CUDAcpl()

    compare("test1_T2",expected,actual)

def test2():
    '''
    permutation
    '''
    a = torch.rand((2,4,1,3,2))
    expected = a.permute((0,2,3,1,4))

    tdd_a = TDD.as_tensor(((a,0,[0,2,3,1]),None))
    actual = TDD.permute(tdd_a, (0,2,3,1)).CUDAcpl()

    compare("test2",expected, actual)



def test3():
    '''
    tracing
    '''
    a = torch.rand((2,2,2,2,2), dtype = torch.double)
    expected = torch.einsum("iijjk->k", a)

    tdd_a = TDD.as_tensor((a,None))
    actual = TDD.trace(tdd_a, [[0,2],[1,3]]).CUDAcpl()

    compare("test3",expected, actual)

def test3_q():
    '''
    tracing (quantum tensors)
    '''
    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmay, CUDAcpl.quantum_basic.sigmax, 0)
    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, a, 0)
    expected = torch.einsum("ijijklm->klm",a)
    
    tdd_a = TDD.as_tensor((a,None))
    actual = TDD.trace(tdd_a, [[0,1],[2,3]]).CUDAcpl()

    compare("test3_q",expected, actual)
    
def test3_q_T():
    '''
    tracing (quantum tensors)
    '''
    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmay, CUDAcpl.quantum_basic.sigmax, 0)
    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, a, 0)
    expected = torch.einsum("klijijm->klm",a)
    
    tdd_a = TDD.as_tensor(((a,2,[]),None))
    actual = TDD.trace(tdd_a, [[0,1],[2,3]]).CUDAcpl()

    compare("test3_q_T",expected, actual)

def test4():
    '''
    large tensor contraction
    '''

    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmay, CUDAcpl.quantum_basic.sigmax, 0)
    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, a, 0)

    b = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, CUDAcpl.quantum_basic.hadamard, 0)
    b = CUDAcpl.tensordot(CUDAcpl.quantum_basic.hadamard, b, 0)

    expected = CUDAcpl.tensordot(a,b,[[1,5,3],[4,0,5]])
    
    tdd_a = TDD.as_tensor(((a,0,[]),None))
    tdd_b = TDD.as_tensor(((b,0,[]),None))
    actual = TDD.tensordot(tdd_a, tdd_b, [[1,5,3],[4,0,5]]).CUDAcpl()

    compare("test4",expected,actual)

def test5():

    '''
    small tensor contraction
    '''

    a = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmay, CUDAcpl.quantum_basic.hadamard, 0)

    b = CUDAcpl.tensordot(CUDAcpl.quantum_basic.sigmax, CUDAcpl.quantum_basic.hadamard, 0)

    expected = CUDAcpl.tensordot(a,b,[[1,3],[3,2]])
    
    tdd_a = TDD.as_tensor(((a,0,[]),None))
    tdd_b = TDD.as_tensor(((b,0,[]),None))
    actual = TDD.tensordot(tdd_a, tdd_b, [[1,3],[3,2]]).CUDAcpl()
    compare("test5",expected,actual)

def test6():

    '''
    micro tensor contraction
    '''

    a = CUDAcpl._U_(torch.rand,(2,2,2))

    b = CUDAcpl.quantum_basic.hadamard

    expected = CUDAcpl.tensordot(a,b,[[0,1],[1,0]])
    
    tdd_a = TDD.as_tensor(((a,0,[]),None))
    tdd_b = TDD.as_tensor(((b,0,[]),None))
    actual = TDD.tensordot(tdd_a, tdd_b, [[0,1],[1,0]]).CUDAcpl()

    compare("test6", expected,actual)

def test7():
    '''
    qft_2 stepwise
    '''

    cz = CUDAcpl.quantum_basic.CZ.reshape((2,2,2,2,2))
    tdd_cz = TDD.as_tensor((cz,None))
    I = CUDAcpl.eye(2)
    tdd_I = TDD.as_tensor((I,None))
    h = CUDAcpl.quantum_basic.hadamard
    tdd_h = TDD.as_tensor((h,None))

    #step 1
    t1 = CUDAcpl.tensordot(h, I, [[0],[1]])
    tdd_t1 = TDD.tensordot(tdd_h, tdd_I, [[0],[1]])
    compare("test7", t1, tdd_t1.CUDAcpl())

    #step 2
    t2 = CUDAcpl.tensordot(I, cz, [[1],[1]])
    tdd_t2 = TDD.tensordot(tdd_I, tdd_cz, [[1],[1]])
    compare("test7", t2, tdd_t2.CUDAcpl())

    #step 3
    t3 = CUDAcpl.tensordot(h, t2, [[0],[3]])
    tdd_t3 = TDD.tensordot(tdd_h, tdd_t2, [[0],[3]])
    compare("test7", t3, tdd_t3.CUDAcpl())

    #step4
    t4 = CUDAcpl.tensordot(t1, t3, [[0],[3]])
    tdd_t4 = TDD.tensordot(tdd_t1, tdd_t3, [[0],[3]])
    compare("test7", t4, tdd_t4.CUDAcpl())

    #permute
    res = t4.permute((0,3,1,2,4))
    tdd_res = TDD.permute(tdd_t4,[0,3,1,2])
    compare("test7", res, tdd_res.CUDAcpl())

def test8():
    '''
    qft_2 stepwise global_order_coordinator version
    '''

    cz = CUDAcpl.quantum_basic.CZ.reshape((2,2,2,2,2))
    tdd_cz = TDD.as_tensor((cz,[2,5,3,6]))
    I = CUDAcpl.eye(2)
    tdd_I1 = TDD.as_tensor((I,[0,1]))
    tdd_I2 = TDD.as_tensor((I,[4,5]))
    h = CUDAcpl.quantum_basic.hadamard
    tdd_h1 = TDD.as_tensor((h,[1,2]))
    tdd_h2 = TDD.as_tensor((h,[6,7]))

    #step 1
    t1 = CUDAcpl.tensordot(h, I, [[0],[1]])
    tdd_t1 = TDD.tensordot(tdd_h1, tdd_I1, [[0],[1]])
    compare("test8", t1, tdd_t1.CUDAcpl())

    #step 2
    t2 = CUDAcpl.tensordot(I, cz, [[1],[1]])
    tdd_t2 = TDD.tensordot(tdd_I2, tdd_cz, [[1],[1]])
    compare("test8", t2, tdd_t2.CUDAcpl())

    #step 3
    t3 = CUDAcpl.tensordot(h, t2, [[0],[3]])
    tdd_t3 = TDD.tensordot(tdd_h2, tdd_t2, [[0],[3]])
    compare("test8", t3, tdd_t3.CUDAcpl())

    #step4
    t4 = CUDAcpl.tensordot(t1, t3, [[0],[3]])
    tdd_t4 = TDD.tensordot(tdd_t1, tdd_t3, [[0],[3]])
    compare("test8", t4, tdd_t4.CUDAcpl())

    #permute
    res = t4.permute((0,3,1,2,4))
    tdd_res = TDD.permute(tdd_t4,[0,3,1,2])
    compare("test8", res, tdd_res.CUDAcpl())

def test9():
    '''
    conj
    '''
    a = torch.rand((2,3,4,2))
    expected = CUDAcpl.np2CUDAcpl(CUDAcpl.CUDAcpl2np(a).conj())

    a_tdd = TDD.as_tensor((a,None))
    actual = TDD.conj(a_tdd).CUDAcpl()

    compare("test9", expected, actual)

def test10():
    '''
    scalar multiply
    '''
    a = CUDAcpl.quantum_basic.CZ;
    expected = CUDAcpl.tensordot(CUDAcpl._U_(torch.tensor, [0,1]), a, 0)

    a_tdd = TDD.as_tensor((a,None))
    actual = TDD.mul(a_tdd, 1j).CUDAcpl()

    compare("test10", expected, actual)

def test1_H():
    '''
    hybrid direct contraction
    '''
    a = CUDAcpl.quantum_basic.sigmax
    b = torch.rand((2,2,2), dtype=torch.double)
    expected = CUDAcpl.einsum("ia,aj->ij",a,b)

    tdd_a = TDD.as_tensor(((a,1,[]),None))
    tdd_b = TDD.as_tensor(((b,0,[]),None))
    actual = TDD.tensordot(tdd_a, tdd_b, [[0],[0]],[], False).CUDAcpl()

    compare("test1_H", expected,actual)
