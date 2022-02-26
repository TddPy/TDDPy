from __future__ import annotations
from typing import TypeVar, List, Any, Sequence, Union

OrderInfo = TypeVar('OrderInfo')

class AbstractCoordinator:

  def __init__(self) -> None:
    self.name = 'abstract coordinator'

  def create_order_info(self, order_info: Any) -> OrderInfo:
    raise NotImplementedError(
        "Coordinator '{}' has not implemented create_order_info.".format(self.name))
    
  def as_tensor_order(self, order_info: OrderInfo)->List[int]:
    raise NotImplementedError(
        "Coordinator '{}' has not implemented as_tensor_order.".format(self.name))


  def trace_order_info(self, order_info: OrderInfo, axes: Sequence[Sequence[int]]) -> OrderInfo:
    raise NotImplementedError("Coordinator '{}' has not implemented trace_order_info.".format(self.name))



  def tensordot_rearrangement(self, info_a: OrderInfo, info_b: OrderInfo, 
                              axes: int|Sequence[Sequence[int]]) -> List[int]:
    raise NotImplementedError(
        "Coordinator '{}' has not implemented tensordot_rearrangement.".format(self.name))

  def tensordot_order_info(self, info_a: OrderInfo, info_b: OrderInfo,
                           axes: int|Sequence[Sequence[int]]) -> OrderInfo:
    raise NotImplementedError(
        "Coordinator '{}' has not implemented tensordot_order_info.".format(self.name))

  def permute_order_info(self, order_info: OrderInfo, perm: Sequence[int]) -> OrderInfo:
    raise NotImplementedError("Coordinator '{}' has not implemented permute_order_info.".format(self.name))

