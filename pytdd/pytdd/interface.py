from __future__ import annotations
from typing import Any, Dict, Tuple, List, Union, Sequence;
import numpy as np
from . import CUDAcpl;
from .CUDAcpl import CUDAcpl_Tensor

# the C++ package
from . import ctdd

# for tdd graphing
from graphviz import Digraph
from IPython.display import Image

TERMINAL_ID = -1

class Node:
    def __init__(self, _pointer):
        self.__pointer : int = _pointer
        if self.pointer != 0:
            self.__info = get_node_info(self)


    @property
    def pointer(self)->int:
        return self.__pointer

    @property
    def info(self) -> Dict:
        '''
            note that terminal node should not use this property
        '''
        if self.pointer == 0:
            raise Exception("terminal node should not use this property")
        return self.__info

    def layout(self, order: List, parallel_shape: Sequence[int], index_order: Sequence[int],
                 dot=Digraph(), succ: List=[], full_output: bool=False, precision: int = 2):
        '''
            full_output: if True, then the edge will appear as a tensor, not the parallel index shape.
        '''


        col=['red','blue','black','green']


        if self.pointer == 0:
            id_str = str(TERMINAL_ID)
            label = str(1)
        else:
            node_info = self.__info
            id_str = str(node_info["id"])
            label = 'i'+str(order[node_info["order"]])


        dot.node(id_str, label, fontname="helvetica",shape="circle",color="red")

        if self.pointer != 0:
            node_successors = node_info["successors"]
            for k in range(node_info["range"]):
                #if there is no parallel index, directly demonstrate the edge values
                if list(node_successors[0]["weight"].shape) == [2]:
                    label1=str(complex(round(node_successors[k]["weight"][0].cpu().item(),precision),
                                        round(node_successors[k]["weight"][1].cpu().item(),precision)))
                #otherwise, demonstrate the parallel index shape
                else:
                    if full_output:
                        label1 = str(CUDAcpl2np(node_successors[k]["weight"]))
                    else:
                        label1 = str(parallel_shape)
                
                temp_node = Node(node_successors[k]["node"])
                if (temp_node.pointer == 0):
                    id_str = str(TERMINAL_ID)
                else:
                    temp_node_info = temp_node.info
                    id_str = str(temp_node_info["id"])
                
                if not temp_node.pointer in succ:
                    dot=temp_node.layout(order, parallel_shape,index_order, dot,succ,full_output)
                    dot.edge(str(node_info["id"]),id_str,color=col[k%4],label=label1)
                    succ.append(temp_node.pointer)
                else:
                    dot.edge(str(node_info["id"]),id_str,color=col[k%4],label=label1)
        return dot    


class TDD:
    def __init__(self, _pointer):
        self.__pointer : int = _pointer
        self.__info = get_tdd_info(self)

    @property
    def pointer(self) -> int:
        return self.__pointer

    @property
    def node(self) -> Node:
        return Node(self.__info["node"])

    @property
    def info(self) -> Dict:
        return self.__info

    @property
    def shape(self) -> Tuple:
        return self.__info["data shape"]
    
    @property
    def index_order(self) -> Tuple:
        return self.__info["index order"]

    # extremely time costy
    def size(self) -> int:
        return ctdd.get_tdd_size(self.__pointer)

    def CUDAcpl(self) -> CUDAcpl_Tensor:
        return to_CUDAcpl(self)

    def numpy(self) -> np.ndarray:
        return to_numpy(self)

    def __str__(self):
        return str(to_numpy(self))


    def show(self, path: str='output', full_output: bool=False, precision: int=2):
        '''
            full_output: if True, then the edge will appear as a tensor, not the parallel index shape.
        '''
        edge=[]              
        tdd_info = self.info
        tdd_node = self.node

        dot=Digraph(name='reduced_tree')
        dot=tdd_node.layout(self.index_order, tdd_info["parallel shape"],tdd_info["index order"], dot, edge, full_output,precision)
        dot.node('-0','',shape='none')

        if tdd_node.pointer == 0:
            id_str = str(TERMINAL_ID)
        else:
            id_str = str(tdd_node.info["id"])

        tdd_weight = tdd_info["weight"]
        if tdd_info["dim parallel"]==0:
            label= str(complex(round(tdd_weight[0].cpu().item(),precision),round(tdd_weight[1].cpu().item(),precision)))
        else:
            if full_output == True:
                label = str(CUDAcpl2np(tdd_weight))
            else:
                label =str(tdd_info["parallel shape"])
        dot.edge('-0',id_str,color="blue",label = label)
        dot.format = 'png'
        return Image(dot.render(path))


    def __del__(self):
        if ctdd.delete_tdd:
            ctdd.delete_tdd(self.__pointer)



#################################################
def delete_tdd(tensor: TDD):
    ctdd.delete_tdd(tensor.pointer)

def reset(device_cuda: bool):
    if device_cuda:
        ctdd.reset(1)
    else:
        ctdd.reset(0)

    
def as_tensor(data : TDD|CUDAcpl_Tensor|np.ndarray|
    Tuple[CUDAcpl_Tensor|np.ndarray, int, Sequence[int]]) -> TDD:
    '''
    construct the tdd tensor

    tensor:
        0. in the form of a TDD, then return a copy of it.
        1. in the form of a matrix only: assume the parallel_index_num to be 0, and index order to be []
        2. in the form of a tuple (data, index_shape, index_order)
        Note that if the input matrix is a torch tensor, 
                then it must be already in CUDAcpl_Tensor(CUDA complex) form.

    '''

    # pre-process
    if isinstance(data,TDD):
        return TDD(ctdd.as_tensor_clone(data.pointer));

    if isinstance(data,Tuple):
        tensor,parallel_i_num,index_order = data
    else:
        tensor = data
        parallel_i_num = 0
        index_order = []

    index_order = list(index_order)
            
    if isinstance(tensor,np.ndarray):
        tensor = CUDAcpl.np2CUDAcpl(tensor)


    # examination

    data_shape = list(tensor.shape[parallel_i_num:-1])

    if len(data_shape)!=len(index_order) and len(index_order)!=0:
        raise Exception('The number of indices must match that provided by tensor.')

    pointer = ctdd.as_tensor(tensor, parallel_i_num, index_order)

    return TDD(pointer)

def to_CUDAcpl(tensor: TDD)->CUDAcpl_Tensor:
    '''
        Transform this tdd to a CUDA complex tensor and return.
    '''
    return ctdd.to_CUDAcpl(tensor.pointer)

def to_numpy(tensor: TDD)->np.ndarray:
    '''
        Transform this tdd to a numpy ndarray.
    '''
    return CUDAcpl.CUDAcpl2np(to_CUDAcpl(tensor))

def trace(tensor: TDD, axes:Sequence[Sequence[int]]) -> TDD:
    '''
        Trace the TDD at given indices.
    '''

    # examination
    if len(axes[0]) != len(axes[1]):
        raise Exception("The indices given by parameter axes does not match.")

    pointer = ctdd.trace(tensor.pointer, axes[0], axes[1])
    return TDD(pointer)

def tensordot(a: TDD, b: TDD,
                axes: int|Sequence[Sequence[int]]) -> TDD:
    '''
        The pytorch-like tensordot method. Note that indices should be counted with data indices only.
        sum_dict_cache: the dictionary cache of former summation calculations.
        parallel_tensor: Whether to tensor on the parallel indices.
    '''
    if isinstance(axes, int):
        pointer = ctdd.tensordot_num(a.pointer, b.pointer, axes)
    else:
        i1 = list(axes[0])
        i2 = list(axes[1])
        if len(i1) != len(i2):
            raise Exception("The list of indices provided")
        pointer = ctdd.tensordot_ls(a.pointer, b.pointer, i1, i2)
    
    res = TDD(pointer)
    return res

def get_tdd_info(tensor: TDD) -> Dict:
    return ctdd.get_tdd_info(tensor.pointer)

def get_tdd_size(tensor: TDD) -> int:
    return ctdd.get_tdd_size(tensor.pointer)

def get_node_info(node: Node) -> Dict:
    return ctdd.get_node_info(node.pointer)

def permute(tensor: TDD, perm: Sequence[int]) -> TDD:
    return TDD(ctdd.permute(tensor.pointer, list(perm)));