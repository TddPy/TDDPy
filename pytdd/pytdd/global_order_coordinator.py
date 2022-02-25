
from pytdd import interface
from typing import Sequence, List
import numpy as np
from .abstract_coordinator import AbstractCoordinator



def order_squeezed(order: Sequence[int]) -> List[int]:
    res = [0]*len(order);
    index = np.argsort(order)
    for i in range(len(order)):
        res[index[i]] = i
    return res

def get_rearrangement(order_a: Sequence[int], order_b: Sequence[int]) -> List[bool]:
    order_all = order_a + order_b
    index = list(range(order_all))
    res = [True] * len(order_a) + [False] * len(order_b)
    sorted(index, lambda i: order_all[i])
    return res[index]

class GlobalOrderCoordinator(AbstractCoordinator):
  def __init__(self) -> None:
    self.name = 'global order coordinator'

    # records the global order of every tensor
    self.tensor_order = dict()

  def as_tensor(self, tensor_data) -> interface.TDD:
    tensor, global_order = tensor_data
    self.tensor_order[]
    order = order_squeezed(self.global_order)

    return interface.as_tensor((tensor_data, 0, order))

  def tensordot(self, tdd_a: interface.TDD, tdd_b: interface.TDD, axes) -> interface.TDD:
    return interface.tensordot(tdd_a, tdd_b, axes)


