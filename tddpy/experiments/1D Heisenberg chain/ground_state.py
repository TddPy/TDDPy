import numpy as np
import CUDAcpl
from CUDAcpl import _U_
import tddpy
from tddpy import TDD
from os import system

import torch




def compare(title, expected: CUDAcpl.CUDAcpl_Tensor,
            actual: CUDAcpl.CUDAcpl_Tensor):
    max_diff = torch.max(abs(expected - actual))

    if ( max_diff > 1e-14):
        print(title+" not passed, diff: ",max_diff)
    else:
        print(title+" passed, diff: ",max_diff)

tddpy.reset(4, False, True, eps = 1E-14)


sx = CUDAcpl.np2CUDAcpl([[0., 0.5],[0.5, 0.]])
sy = CUDAcpl.np2CUDAcpl([[0., -0.5j],[0.5j, 0.]])
sz = CUDAcpl.np2CUDAcpl([[0.5, 0.],[0., -0.5]])

tdd_sx = TDD.as_tensor((sx, None))
tdd_sy = TDD.as_tensor((sy, None))
tdd_sz = TDD.as_tensor((sz, None))

tdd_sxsx = TDD.tensordot(tdd_sx,tdd_sx,0)
tdd_sysy = TDD.tensordot(tdd_sy,tdd_sy,0)
tdd_szsz = TDD.tensordot(tdd_sz,tdd_sz,0)

tdd_h = (tdd_sxsx + tdd_sysy) + tdd_szsz

def normalize(tensor):
    '''
        return the normalized version of the given tensor as a vector
    '''
    tensor_conj = TDD.conj(tensor)
    ils = list(range(len(tensor.shape)))
    norm = CUDAcpl.CUDAcpl2np(TDD.tensordot(tensor_conj, tensor, [ils,ils]).CUDAcpl())
    print("current norm: ", norm)
    norm_sqrt = np.sqrt(norm)
    return TDD.mul(tensor, 1/norm_sqrt)
    

def rand_init_state(n):
    '''
        produce the randomized initial state
    '''
    res = TDD.as_tensor((_U_(torch.rand, (2,2)), None))

    stuff_tensor = TDD.as_tensor((_U_(torch.tensor, [[1.,0.],[1.,0.]]),None))


    for i in range(1, n):
        res = TDD.tensordot(res, stuff_tensor, 0)
    return normalize(res)
    
def hamil_mul_vec(n, vec):
    # periodic boundary condition
    res = TDD.tensordot(tdd_h, vec, [[3, 1],[0, n-1]])

    for i in range(0, n-1):
        temp = TDD.tensordot(tdd_h, vec, [[1, 3], [i, i+1]])
        res = res + temp
    
    return res



if __name__ == "__main__":
    dt = 0.01
    n = 18
    state = rand_init_state(n)

    norm_period = 10

    for i in range(5000):
        print(i, "...")

        # tdd backend
        next_state = state + TDD.mul(hamil_mul_vec(n, state), -dt+0.j)
        if i % norm_period == 0:
            next_state = normalize(next_state)

        
            next_state_conj = TDD.conj(next_state)
            ils = list(range(len(next_state.shape)))
            norm = CUDAcpl.CUDAcpl2np(TDD.tensordot(next_state_conj, state, [ils,ils]).CUDAcpl())

            print("1 - fidelity: ", 1 - norm)
            tddpy.clear_cache()

        state = next_state
        print(state.size())
