
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Union, Sequence;
import numpy as np
from . import CUDAcpl;
from .CUDAcpl import CplTensor, CUDAcpl2np

# the C++ package
from . import ctdd

# for tdd graphing
from graphviz import Digraph
from IPython.display import Image

TERMINAL_ID = -1

class Node:
    def get_node_info(self):
        if self.__tensor_weight:
            return ctdd.get_node_info_T(self.__pointer)
        else:
            return ctdd.get_node_info(self.__pointer)

    def __init__(self, _pointer, _tensor_weight : bool):
        self.__pointer : int = _pointer
        self.__tensor_weight : bool = _tensor_weight
        if self.pointer != 0:
            self.__info = self.get_node_info()

    @property
    def tensor_weight(self)->bool:
        return self.__tensor_weight

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

    @property
    def range(self) -> int:
        return self.__info["range"]

    @property
    def order(self) -> int:
        return self.__info["order"]

    def layout(self, storage_order: Sequence[int], parallel_shape: Sequence[int],
                 dot=Digraph(), succ: List=[], full_output: bool=False, precision: int = 2, tensor_weight: bool = False):
        '''
            full_output: if True, then the edge weight will appear.
        '''


        col=['red','blue','black','green']


        if self.pointer == 0:
            id_str = str(TERMINAL_ID)
            label = str(1)
        else:
            node_info = self.__info
            id_str = str(self.__pointer)
            label = 'i'+str(storage_order[self.order])


        dot.node(id_str, label, fontname="helvetica",shape="circle",color="red")

        if self.pointer != 0:
            node_successors = node_info["successors"]
            for k in range(self.range):
                #if there is no parallel index, directly demonstrate the edge values
                if full_output:
                    if tensor_weight:
                        label_weight = str(CUDAcpl2np(node_successors[k]["weight"]))
                    else:
                        label_weight=str(complex(round(node_successors[k]["weight"][0].cpu().item(),precision),
                                            round(node_successors[k]["weight"][1].cpu().item(),precision)))
                else:
                    label_weight = ""
                
                temp_node = Node(node_successors[k]["node"], self.tensor_weight)
                if (temp_node.pointer == 0):
                    id_str = str(TERMINAL_ID)
                else:
                    id_str = str(temp_node.__pointer)
                
                if not temp_node.pointer in succ:
                    dot=temp_node.layout(storage_order, parallel_shape, dot,succ,full_output, precision, tensor_weight)
                    dot.edge(str(self.__pointer),id_str,color=col[k%4],label=label_weight)
                    succ.append(temp_node.pointer)
                else:
                    dot.edge(str(self.__pointer),id_str,color=col[k%4],label=label_weight)
        return dot    