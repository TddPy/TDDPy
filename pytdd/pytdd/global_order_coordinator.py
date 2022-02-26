from __future__ import annotations
from typing import List, Any, Sequence, Union
import numpy as np

from .abstract_coordinator import OrderInfo, AbstractCoordinator



def order_squeezed(order: Sequence[int]) -> List[int]:
    res = [0]*len(order);
    index = np.argsort(order)
    for i in range(len(order)):
        res[index[i]] = i
    return res

class GlobalOrderCoordinator(AbstractCoordinator):

  def __init__(self) -> None:
    self.name = 'trival coordinator'

  def create_order_info(self, order_info: Any) -> OrderInfo:
      '''
        here the order_info should be the corresponding orders of tensor indices, in the global index order
      '''
      if order_info == None:
          return None
      return list(order_info)

  def as_tensor_order(self, order_info: OrderInfo)->List[int]:
      if order_info == None:
          return []

      temp = order_squeezed(order_info)
      # a reverse is needed
      res = [0] * len(temp)
      for i in range(len(temp)):
          res[temp[i]] = i
      return res

  def trace_order_info(self, order_info: OrderInfo, axes: Sequence[Sequence[int]]) -> OrderInfo:
      if order_info == None:
          return None

      res = []
      for i in range(len(order_info)):
          if i not in axes[0] and i not in axes[1]:
              res.append(order_info[i])
      return res


  def tensordot_rearrangement(self, info_a: OrderInfo, info_b: OrderInfo, 
                              axes: int|Sequence[Sequence[int]]) -> List[int]:
      if info_a == None or info_b == None:
          return []

      if isinstance(axes, int):
          num = axes
      else:
          num = len(axes[0])

      order_all = self.tensordot_order_info(info_a, info_b, axes)

      index = list(range(len(order_all)))
      res = [1] * (len(info_a) - num) + [0] * (len(info_b) - num)
      index = sorted(index, key = lambda i: order_all[i])
      res = [res[i] for i in index]
      return res

  def tensordot_order_info(self, info_a: OrderInfo, info_b: OrderInfo,
                           axes: int|Sequence[Sequence[int]]) -> OrderInfo:
      '''
        The order after tensordot is determined by the standard order rearrangement of tensordot
      '''
      if info_a == None or info_b == None:
          return None

      if isinstance(axes, int):
          num = axes
          temp1 = list(range(len(info_a)-num, len(info_a)))
          temp2 = list(range(num))
          axes = [temp1, temp2]

      order_all=[] 
      for i in range(len(info_a)):
          if i not in axes[0]:
              order_all.append(info_a[i])
      for i in range(len(info_b)):
          if i not in axes[1]:
              order_all.append(info_b[i])

      return order_all

  def permute_order_info(self, order_info: OrderInfo, perm: Sequence[int]) -> OrderInfo:
      if order_info == None:
          return None

      res = [0]*len(order_info)
      for i in range(len(order_info)):
          res[i] = order_info[perm[i]]

      return res

