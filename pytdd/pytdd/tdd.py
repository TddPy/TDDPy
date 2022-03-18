
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Union, Sequence;
import numpy as np
from . import CUDAcpl;
from .CUDAcpl import CUDAcpl_Tensor, CUDAcpl2np

# the C++ package
from . import ctdd

# the TDD index node
from .node import Node

# for tdd graphing
from graphviz import Digraph
from IPython.display import Image

TERMINAL_ID = -1

class TDD:


    # different invocations for scalar and tensor weight
    def get_tdd_info(self):
        if self._tensor_weight:
            return ctdd.get_tdd_info_T(self._pointer)
        else:
            return ctdd.get_tdd_info(self.pointer)


    def __init__(self, pointer, tensor_weight: bool):
        self._pointer : int = pointer
        self._tensor_weight = tensor_weight
        self._info = self.get_tdd_info()


    @property
    def tensor_weight(self)->bool:
        return self._tensor_weight

    @property
    def pointer(self) -> int:
        return self._pointer

    @property
    def node(self) -> Node:
        return Node(self._info["node"], self._tensor_weight)

    @property
    def info(self) -> Dict:
        return self._info

    @property
    def shape(self) -> Tuple:
        return self._info["data shape"]
    
    @property
    def parallel_shape(self) -> Tuple:
        return self._info["parallel shape"]

    @property
    def storage_order(self) -> Tuple:
        return self._info["storage order"]

    def size(self) -> int:
        if self._tensor_weight:
            return ctdd.get_tdd_size_T(self._pointer)
        else:
            return ctdd.get_tdd_size(self._pointer)

    def CUDAcpl(self) -> CUDAcpl_Tensor:
        if self._tensor_weight:
            return ctdd.to_CUDAcpl_T(self._pointer)
        else:
            return ctdd.to_CUDAcpl(self._pointer)

    def numpy(self) -> np.ndarray:
        return CUDAcpl.CUDAcpl2np(self.CUDAcpl())

    def __str__(self):
        return str(self.numpy())


    def show(self, path: str='output', full_output: bool=False, precision: int=2):
        '''
            full_output: if True, then the edge will appear as a tensor, not the parallel index shape.
        '''
        edge=[]              
        tdd_node = self.node

        dot=Digraph(name='reduced_tree')
        dot=tdd_node.layout(self.storage_order, self.parallel_shape, dot, edge, full_output,precision)
        dot.node('-0','',shape='none')

        if tdd_node.pointer == 0:
            id_str = str(TERMINAL_ID)
        else:
            id_str = str(tdd_node.pointer)

        tdd_weight = self.info["weight"]
        if self.info["dim parallel"]==0:
            label= str(complex(tdd_weight[0].cpu().item(),tdd_weight[1].cpu().item()))
        else:
            if full_output == True:
                label = str(CUDAcpl2np(tdd_weight))
            else:
                label =str(self.parallel_shape)
        dot.edge('-0',id_str,color="blue",label = 
                 "paralell shape: " +str(self.parallel_shape) + 
                 "\ndata shape:" + str(self.shape) + "\n" + label 
                 )
        dot.format = 'png'
        return Image(dot.render(path))


    def __del__(self):
        if ctdd:
            if self._tensor_weight:
                if ctdd.delete_tdd_T:
                    ctdd.delete_tdd_T(self._pointer)
            else:
                if ctdd.delete_tdd:
                    ctdd.delete_tdd(self._pointer)


    # the tensor methods

    @staticmethod
    def as_tensor(data : TDD|
                      CUDAcpl_Tensor|np.ndarray|Tuple[CUDAcpl_Tensor|np.ndarray, int, Sequence[int]]) -> TDD:

        '''
        construct the tdd tensor

        data:
            0. in the form of a TDD, then return a copy of it.
            1. in the form of (tensor, coordinator_info), where tensor is
                1a. in the form of a matrix only: assume the parallel_index_num to be 0, and index order to be [],
                    <return a scalar weight TDD>
                1b. in the form of a tuple (data, index_shape, index_order)
                    <if index_shape == 0, then return a scalar weight TDD, else return a tensor weight TDD>
                Note that if the input matrix is a torch tensor, 
                        then it must be already in CUDAcpl_Tensor(CUDA complex) form.

        '''

        # pre-process
        if isinstance(data, TDD):
            # note the order information is also copied
            if data._tensor_weight:
                return TDD(ctdd.as_tensor_clone_T(data.pointer), True);
            else:
                return TDD(ctdd.as_tensor_clone(data.pointer), False);

        if isinstance(data,Tuple):
            tensor,parallel_i_num,storage_order = data
        else:
            tensor = data
            parallel_i_num = 0
            storage_order = []
            
        if isinstance(tensor,np.ndarray):
            tensor = CUDAcpl.np2CUDAcpl(tensor)

        # examination

        data_shape = list(tensor.shape[parallel_i_num:-1])

        if len(data_shape)!=len(storage_order) and len(storage_order)!=0:
            raise Exception('The number of indices must match that provided by tensor.')

        tensor_weight = (parallel_i_num != 0)

        if tensor_weight:
            pointer = ctdd.as_tensor_T(tensor, parallel_i_num, storage_order)
        else:
            pointer = ctdd.as_tensor(tensor, 0, storage_order)

        return TDD(pointer, tensor_weight)

    @staticmethod
    def conj(tensor: TDD) -> TDD:
        '''
            Return the conjugate of the tdd tensor.
            # note that the coordinator information is not changed.
        '''
        if tensor._tensor_weight:
            pointer = ctdd.conj_T(tensor.pointer)
        else:
            pointer = ctdd.conj(tensor.pointer)

        return TDD(pointer, tensor.tensor_weight)

    @staticmethod
    def mul(tensor: TDD, scalar: CUDAcpl_Tensor|complex) -> TDD:
        '''
            Return the tdd multiplied by the scalar (tensor).
            Note that the coordinator information will not be changed.
        '''
        if tensor._tensor_weight:
            if isinstance(scalar, complex):
                pointer = ctdd.mul_TW(tensor.pointer, scalar)
            elif isinstance (scalar, CUDAcpl_Tensor):
                pointer = ctdd.mul_TT(tensor.pointer, scalar)
            else:
                raise "The scalar must be a python complex or a CUDAcpl tensor."
        else:
            if isinstance(scalar, complex):
                pointer = ctdd.mul_WW(tensor.pointer, scalar)
            else:
                raise "The scalar must be a python complex for this scalar weight tdd."

        return TDD(pointer, tensor._tensor_weight)

    def __add__(self, other: TDD) -> TDD:
        '''
            return the summation of two tdds
            Note that the coordinator information is not changed.
        '''

        if self.storage_order != other.storage_order \
            or self.parallel_shape != other.parallel_shape \
            or self.tensor_weight != other.tensor_weight:
            raise "Only two tdds of the same storage order and the same parallel shape can be summed up."
        if self.tensor_weight:
            pointer = ctdd.sum_T(self.pointer, other.pointer)
        else:
            pointer = ctdd.sum_W(self.pointer, other.pointer)

        return TDD(pointer, self._tensor_weight)

    @staticmethod
    def trace(tensor: TDD, axes:Sequence[Sequence[int]]) -> TDD:
        '''
            Trace the TDD at given indices.
        '''
        # examination
        if len(axes[0]) != len(axes[1]):
            raise Exception("The indices given by parameter axes does not match.")

        if tensor.tensor_weight:
            pointer = ctdd.trace_T(tensor.pointer, list(axes[0]), list(axes[1]))
        else:
            pointer = ctdd.trace(tensor.pointer, list(axes[0]), list(axes[1]))

        return TDD(pointer, tensor.tensor_weight)


    @staticmethod
    def tensordot(a: TDD, b: TDD, 
                  axes: int|Sequence[Sequence[int]], rearrangement: Sequence[bool] = [],
                  parallel_tensor: bool = False) -> TDD:
        
        '''
            The pytorch-like tensordot method. Note that indices should be counted with data indices only.
            rearrangement: If not [], then will rearrange according to the parameter. Otherwise, it will rearrange according to the coordinator.
            parallel_tensor: Whether to tensor on the parallel indices.
        '''
        parallel_tensor = 1 if parallel_tensor else 0

        if isinstance(axes, int):
            # conditioning on the weight version and iteration parallelism
            if not a.tensor_weight and not b.tensor_weight:
                pointer = ctdd.tensordot_num_WW(a.pointer, b.pointer, axes, rearrangement, parallel_tensor)
                res_tensor_weight = False
            elif a.tensor_weight and b.tensor_weight:
                pointer = ctdd.tensordot_num_TT(a.pointer, b.pointer, axes, rearrangement, parallel_tensor)
                res_tensor_weight = True
            elif a.tensor_weight and not b.tensor_weight:
                pointer = ctdd.tensordot_num_TW(a.pointer, b.pointer, axes, rearrangement, parallel_tensor)
                res_tensor_weight = True
            else:
                pointer = ctdd.tensordot_num_WT(a.pointer, b.pointer, axes, rearrangement, parallel_tensor)
                res_tensor_weight = True

        else:
            i1 = list(axes[0])
            i2 = list(axes[1])
            if len(i1) != len(i2):
                raise Exception("The list of indices provided")
            
            # conditioning on the weight version and iteration parallelism
            if not a.tensor_weight and not b.tensor_weight:
                pointer = ctdd.tensordot_ls_WW(a.pointer, b.pointer, i1, i2, rearrangement, parallel_tensor)
                res_tensor_weight = False
            elif a.tensor_weight and b.tensor_weight:
                pointer = ctdd.tensordot_ls_TT(a.pointer, b.pointer, i1, i2, rearrangement, parallel_tensor)
                res_tensor_weight = True
            elif a.tensor_weight and not b.tensor_weight:
                pointer = ctdd.tensordot_ls_TW(a.pointer, b.pointer, i1, i2, rearrangement, parallel_tensor)
                res_tensor_weight = True
            else:
                pointer = ctdd.tensordot_ls_WT(a.pointer, b.pointer, i1, i2, rearrangement, parallel_tensor)
                res_tensor_weight = True
        
        res = TDD(pointer, res_tensor_weight)
        return res


    @staticmethod
    def permute(tensor: TDD, perm: Sequence[int]) -> TDD:
        if tensor.tensor_weight:
            return TDD(ctdd.permute_T(tensor.pointer, list(perm)), True);
        else:
            return TDD(ctdd.permute(tensor.pointer, list(perm)), False);
