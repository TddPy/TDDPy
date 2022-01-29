from __future__ import annotations
from ast import Index
from typing import Iterable, Tuple, List, Any, Union, cast
from xmlrpc.client import Boolean
import numpy as np
import torch

from . import CUDAcpl
from .CUDAcpl import _U_, CUDAcpl_Tensor
from . import node
from .node import  Node
import copy


from graphviz import Digraph
from IPython.display import Image


IndexOrder = List[int]

class TDD:
    '''
        TDD  functions as the compact representation of tensors,
        and can fit into tensor networks.
    '''
    def __init__(self, 
                    weights: CUDAcpl_Tensor,
                    parallel_shape: List[int],
                    data_shape: List[int],
                    node: Node,
                    index_order: IndexOrder = []):
        self.weights: CUDAcpl_Tensor = weights
        self.parallel_shape: List[int] = parallel_shape
        self.data_shape: List[int] = data_shape  #the data index shape
        self.node: Node = node

        '''
            index_order: how the inner index are mapped to outer representations
            for example, tdd[a,b,c] under index_order=[0,2,1] returns the value tdd[a,c,b]
            index_order == None means the trival index mapping [0,1,2,(...)]
        '''
        self.index_order: IndexOrder = index_order

    def __eq__(self, other: TDD) -> Boolean:
        '''
            Now this equality check only deals with TDDs with the same index order.
        '''
        if self.index_order==other.index_order:
            if self.node == other.node \
                and (Node.get_int_key(self.weights)==Node.get_int_key(other.weights)).all():
                return True
            else:
                return False
        else:
            raise Exception('index order not the same!')

    @staticmethod
    def __construct_and_normalize(parallel_shape: List[int], the_successors: List[TDD]):
        '''
            construct the tdd with the_successors, and normalize it
            Note: This method requires the index order and data shape of the_successors to be the same.
                    Global information like index order and data shape are not generated.
        '''
        all_equal = True
        for k in range(1,len(the_successors)):
            if the_successors[k] != the_successors[0]:
                all_equal = False
                break
        if all_equal:
            return the_successors[0]
        
        #pay attention that out_weigs are stacked at the first index here
        weigs=torch.stack([succ.weights for succ in the_successors])

        all_zeros = True
        #this zero-checking is not written in CUDA because we expect it to ternimate very soon
        for k in range(len(the_successors)):
            int_key = Node.get_int_key(weigs[k])
            if torch.max(int_key) != 0 or torch.min(int_key) != 0:
                all_zeros=False
                break
        if all_zeros:
            weights=_U_(torch.zeros_like,the_successors[0].weights)
            
            node=Node.get_terminal_node()

            res=TDD(weights,parallel_shape,[],node,[])
            return res

        for k in range(len(the_successors)):
            if the_successors[k].data_shape != the_successors[0].data_shape:
                raise Exception('This method requires the data shape of the_successors to be the same.')
            if the_successors[k].index_order != the_successors[0].index_order:
                raise Exception('This method requires the index order of the_successors to be the same.')

            int_key = Node.get_int_key(weigs[k])
            if torch.max(int_key) == 0 and torch.min(int_key) == 0:
                weights = _U_(torch.zeros_like,weigs[k])
                node=Node.get_terminal_node()
                the_successors[k]=TDD(weights,parallel_shape,[],node,[])

                weigs[k] = _U_(torch.zeros_like,weigs[k])
        
        weig_abs = CUDAcpl.norm(weigs)
        max_indices = torch.max(weig_abs,dim=0,keepdim=True).indices
        weig_max = torch.stack((weigs[...,0].gather(-1,max_indices),weigs[...,1].gather(-1,max_indices)),-1)[0] #shape: [?,2]
        weig_max_1 = torch.stack((weig_max[...,0],-weig_max[...,1]),-1)/(weig_max[...,0]**2+weig_max[...,1]**2)
        weigs = CUDAcpl.einsum('k...,...->k...',weigs,weig_max_1)   #shape: [weigs_index,?,2],[?,2]->[weigs_index,?,2]
        succ_nodes=[succ.node for succ in the_successors]

        node=Node.get_unique_node(weigs,succ_nodes)
        res=TDD(weig_max,parallel_shape,[],node,[])
        return res



    @staticmethod
    def __as_tensor_iterate(tensor : CUDAcpl_Tensor, 
                    parallel_shape: List[int],
                    index_order: List[int], depth: int) -> TDD:
        '''
            The inner interation for as_tensor.
            depth: current iteration depth, used to indicate index_order and termination

            Guarantee: parallel_shape and index_order will not be modified.
        '''

        data_shape = list(tensor.shape[len(parallel_shape):-1])  #the data index shape
        if index_order == []:
            index_order = list(range(len(data_shape)))

        #checks whether the tensor is reduced to the [[...[val]...]] form
        if depth == len(data_shape):

            #maybe some improvement is needed here.
            if len(data_shape)==0:
                weights = tensor.clone()
            else:
                weights = (tensor[...,0:1,:]).clone().detach().view(parallel_shape+[2])
            node = Node.get_terminal_node()
            res = TDD(weights,parallel_shape,[],node,[])
            return res
        

        split_pos=index_order[depth]
        split_tensor = list(tensor.split(1,-len(data_shape)+split_pos-1))
            #-1 is because the extra inner dim for real and imag

        the_successors: List[TDD] =[]

        for k in range(data_shape[split_pos]):
            res = TDD.__as_tensor_iterate(split_tensor[k],parallel_shape,index_order,depth+1)
            the_successors.append(res)
        
        tdd = TDD.__construct_and_normalize(parallel_shape,the_successors)
        return tdd


    @staticmethod
    def as_tensor(tensor : CUDAcpl_Tensor, 
                    parallel_shape: List[int] = [],
                    index_order: IndexOrder = []) -> TDD:


        data_shape = list(tensor.shape[len(parallel_shape):-1])  #the data index shape
        if index_order == []:
            result_index_order = list(range(len(data_shape)))
        else:
            result_index_order = index_order.copy()


        if len(data_shape)!=len(index_order):
            raise Exception('The number of indices must match that provided by tensor.')

        '''
            This extra layer is for copying the input list and pre-process.
        '''
        res = TDD.__as_tensor_iterate(tensor,parallel_shape,index_order,0)

        
        res.index_order = result_index_order
        res.data_shape = data_shape
        return res



    def show(self,real_label=True,path: str='output'):
        edge=[]              
        dot=Digraph(name='reduced_tree')
        dot=self.node.layout(self.index_order,dot,edge,real_label)
        dot.node('-0','',shape='none')
        if list(self.weights.shape)==[2]:
            dot.edge('-0',str(self.node.id),color="blue",label=str(complex(round(self.weights[0].cpu().item(),2),round(self.weights[1].cpu().item(),2))))
        else:
            dot.edge('-0',str(self.node.id),color="blue",label=str(self.parallel_shape))
        dot.format = 'png'
        return Image(dot.render(path))