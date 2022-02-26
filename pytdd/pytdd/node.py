
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Union, Sequence;
import numpy as np
from . import CUDAcpl;
from .CUDAcpl import CUDAcpl_Tensor, CUDAcpl2np

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
            self.__info = ctdd.get_node_info(self.__pointer)


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
    def id(self) -> int:
        return self.__info["id"]

    @property
    def range(self) -> int:
        return self.__info["range"]

    @property
    def order(self) -> int:
        return self.__info["order"]

    def layout(self, storage_order: Sequence[int], parallel_shape: Sequence[int],
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
            id_str = str(self.id)
            label = 'i'+str(storage_order[self.order])


        dot.node(id_str, label, fontname="helvetica",shape="circle",color="red")

        if self.pointer != 0:
            node_successors = node_info["successors"]
            for k in range(self.range):
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
                    id_str = str(temp_node.id)
                
                if not temp_node.pointer in succ:
                    dot=temp_node.layout(storage_order, parallel_shape, dot,succ,full_output)
                    dot.edge(str(self.id),id_str,color=col[k%4],label=label1)
                    succ.append(temp_node.pointer)
                else:
                    dot.edge(str(self.id),id_str,color=col[k%4],label=label1)
        return dot    