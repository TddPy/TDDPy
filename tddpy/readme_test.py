import torch
import tddpy
from tddpy.CUDAcpl import quantum_circ

h_tdd = tddpy.TDD.as_tensor(quantum_circ.hadamard())
theta = torch.tensor([0.1, 0.2, 0.3], dtype = torch.double)
layer1 = quantum_circ.Rx(theta)
layer1_tdd = tddpy.TDD.as_tensor((layer1, 1, []))
res_cont = tddpy.TDD.tensordot(h_tdd, layer1_tdd, [[1],[0]])
print(res_cont.numpy())
res_cont.show("hybrid_cont_res")