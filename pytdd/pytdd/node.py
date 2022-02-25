
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