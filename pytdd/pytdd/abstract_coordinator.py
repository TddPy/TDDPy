from __future__ import annotations
from typing import TypeVar, List, Any, Sequence, Union
from .tdd import TDD

OrderInfo = TypeVar('OrderInfo')

class WrappedTDD:
	def __init__(self, tensor: TDD, crd_info: OrderInfo) -> WrappedTDD:
		self.crd_info = crd_info
		self.tensor = tensor

	@property
	def shape(self):
		return self.tensor.shape

	def CUDAcpl(self):
		return self.tensor.CUDAcpl()

	def numpy(self):
		return self.tensor.numpy()
	
	def size(self):
		return self.tensor.size()

	@property
	def info(self):
		return self.tensor.info


class AbstractCoordinator:

	def __init__(self):
		self.name = "Abstract Coordinator"


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




	# the methods

	def as_tensor(self, wrapped_data : WrappedTDD|
				  Tuple[
					  CUDAcpl_Tensor|np.ndarray|Tuple[CUDAcpl_Tensor|np.ndarray, int, Sequence[int]],
					  Any
				  ]) -> WrappedTDD:

		if isinstance(wrapped_data, WrappedTDD):
			# note the order information is also copied
			return WrappedTDD(TDD.as_tensor(wrapped_data.tensor), wrapped_data.crd_info)

		# extract the order information
		data, coordinator_info = wrapped_data
		coordinator_info = self.create_order_info(coordinator_info)

		if isinstance(data, tuple):
			tensor, parallel_i_num, storage_order = data
		else:
			tensor = data
			parallel_i_num = 0
			storage_order = []

		if storage_order == []:
			storage_order = self.as_tensor_order(coordinator_info)

		return WrappedTDD(TDD.as_tensor((tensor, parallel_i_num, storage_order)),
							coordinator_info)


	def trace(self, wrapped_tdd: WrappedTDD, axes:Sequence[Sequence[int]]) -> WrappedTDD:

		return WrappedTDD(TDD.trace(wrapped_tdd.tensor, axes), 
							 self.trace_order_info(wrapped_tdd.crd_info))


	def tensordot(self, wrapped_a: WrappedTDD, wrapped_b: WrappedTDD, 
				  axes: int|Sequence[Sequence[int]], rearrangement: Sequence[bool] = [],
				  parallel_tensor: bool = False, iteration_parallel: bool = True) -> WrappedTDD:
		
		# need to check whether a and b are the same coordinator

		if rearrangement == []:
			rearrangement = self.tensordot_rearrangement(wrapped_a.crd_info, wrapped_b.crd_info, axes)

		new_coordinator_info = self.tensordot_order_info(wrapped_a.crd_info, wrapped_b.crd_info, axes)

		return WrappedTDD(
			TDD.tensordot(wrapped_a.tensor, wrapped_b.tensor, axes, rearrangement, parallel_tensor, iteration_parallel),
			new_coordinator_info)


	def permute(self, wrapped_tdd: WrappedTDD, perm: Sequence[int]) -> WrappedTDD:
		return WrappedTDD(TDD.permute(wrapped_tdd.tensor, perm), 
							 self.permute_order_info(wrapped_tdd.crd_info, perm))