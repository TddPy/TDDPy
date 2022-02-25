from __future__ import annotations
from typing import List, Any, Sequence, Union

from .abstract_coordinator import OrderInfo, AbstractCoordinator

class TrivalCoordinator(AbstractCoordinator):

  def __init__(self) -> None:
    self.name = 'trival coordinator'

  def create_order_info(self, order_info: Any) -> OrderInfo:
      return None

  def as_tensor_order(self, order_info: OrderInfo)->List[int]:
      return []

  def trace_order_info(self, order_info: OrderInfo, axes: Sequence[Sequence[int]]) -> OrderInfo:
      return None


  def tensordot_rearrangement(self, info_a: OrderInfo, info_b: OrderInfo, 
                              axes: int|Sequence[Sequence[int]]) -> List[bool]:
      return []

  def tensordot_order_info(self, info_a: OrderInfo, info_b: OrderInfo,
                           axes: int|Sequence[Sequence[int]]) -> OrderInfo:
      return None

  def permute_order_info(self, order_info: OrderInfo, perm: Sequence[int]) -> OrderInfo:
      return None

