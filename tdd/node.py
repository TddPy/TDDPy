from __future__ import annotations
from ast import Index
from typing import List, Tuple, Union, cast
from . import CUDAcpl
from .CUDAcpl import CUDAcpl_Tensor,_U_,CUDAcpl2np
import torch

from graphviz import Digraph

TERMINAL_KEY = 0

IndexOrder = List[int]

def order_inverse(index_order: IndexOrder) -> IndexOrder:
    '''
        Return the "inverse" of the given index order.
        (it can be understand as the inverse in the permutation group.)
    '''
    res = [0]*len(index_order)

    for i in range(len(index_order)):
        res[index_order[i]] = i
    
    return res

class Node:
    '''
        The node used in tdd.
    '''
    
    
    EPS=0.000001
    '''
        The precision for comparing two float numbers.
        It also decides the precision of weights stored in unique_table.
    '''


    @staticmethod
    def get_int_key_long(weight: CUDAcpl_Tensor):
        return torch.round(weight/Node.EPS).long()

    @staticmethod
    def get_int_key_int(weight: CUDAcpl_Tensor):
        return torch.round(weight/Node.EPS).int()

    @staticmethod
    def get_int_key_short(weight: CUDAcpl_Tensor):
        return torch.round(weight/Node.EPS).short()

    
    get_int_key = get_int_key_long  #this function takes in the weight tensor and generates the integer key for unique_table


    __unique_table = dict() 
    '''
        The unique_table to store all the node instances used in tdd.
        dictionary key:
            TERMINAL_KEY for terminal node
            [order, index(weight1,weight2...), successor1, successor2,...] for non-terminal nodes
    '''
    global_node_id = 0 #it counts the total number of nonterminal nodes

    @staticmethod
    def reset():
        Node.__unique_table.clear()
        Node.global_node_id = 0
        
        terminal_node = Node()
        terminal_node.id = 0
        terminal_node.order = -1
        terminal_node.out_weights = _U_(torch.tensor,[])
        terminal_node.successors = []

        Node.__unique_table[TERMINAL_KEY] = terminal_node


    def __init__(self):
        '''
        The structure of node instances:
        - id
        - order : represent the order of this node (which tensor index it represent)
        - out_weights : torch.Tensor, shape: [succ_num, ..., 2]. The first index is for the successors,
                        and the last index is for complex representation.
        - successor
        '''
        self.id : int = 0
        self.order : int = 0
        self.out_weights : CUDAcpl_Tensor = _U_(torch.tensor,[])
        self.successors : List[Node] = []

    @property
    def index_range(self) -> int:
        # how many values this index can take
        return len(self.successors)

    @staticmethod
    def get_terminal_node():
        return Node.__unique_table[TERMINAL_KEY]

    @staticmethod
    def get_unique_node(order:int, out_weights: CUDAcpl_Tensor, succ_nodes: List[Node]) -> Node:
        '''
            Return the required node. It is either from the unique table, or a newly created one.
            
            order: represent the order of this node (which tensor index it represent)
            out_weights: the incoming weights of this node, shape: [succ_num, ..., 2].
            succ_nodes: the successor nodes.

            Note: The equality checking inside is conducted with the node.EPS tolerance. So feel free
                    to pass in the raw weights from calculation.
        '''
                
        #generate the unique key
        temp_key : Tuple[int|Node]
        temp_key = tuple(cast(List[Union[int,Node]],[order]) 
                    + cast(List[Union[int,Node]],Node.get_int_key(out_weights).view(-1).tolist()) 
                    + cast(List[Union[int,Node]],succ_nodes))

        if temp_key in Node.__unique_table:
            return Node.__unique_table[temp_key]
        else:
            res = Node()
            Node.global_node_id += 1
            res.id = Node.global_node_id
            res.order = order
            res.out_weights = out_weights.clone().detach()
            res.successors = succ_nodes.copy()
            Node.__unique_table[temp_key] = res
            return res
    
    def CUDAcpl_Tensor(self, weights: CUDAcpl_Tensor, data_shape: Tuple[int]) -> CUDAcpl_Tensor:
        '''
            Get the CUDAcpl_Tensor determined from this node and the weights.

            (use the trival index order)
            data_shape(in the corresponding trival index order) is required to broadcast at reduced nodes of indices.
        '''

        parallel_shape = tuple(weights.shape[:-1])

        if self.id == TERMINAL_KEY:
            res = CUDAcpl.ones(parallel_shape)
        else:
            res = self.__CUDAcpl_Tensor(weights, data_shape)
        
        #this extra layer is for adding the reduced dimensions at the front
        res = res.view(parallel_shape+self.order*(1,)+res.shape[len(parallel_shape):])
        res = res.expand(parallel_shape + data_shape+(2,))

        return res
        




    def __CUDAcpl_Tensor(self, weights: CUDAcpl_Tensor, data_shape: Tuple[int]) -> CUDAcpl_Tensor:
        '''
            Note: due to the special handling of iteration, this method expect 'self' to be a non-terminal node.
        '''
        current_order = self.order

        parallel_shape = tuple(weights.shape[:-1])
        par_tensor = []
        for k in range(self.index_range):
            #detect terminal nodes, or iterate on the next node
            if self.successors[k].id == TERMINAL_KEY:
                temp_tensor = self.out_weights[k]
                next_order = len(data_shape)
            else:
                next_order = self.successors[k].order
                temp_tensor = self.successors[k].__CUDAcpl_Tensor(self.out_weights[k],data_shape)
                expanded_out_weights = self.out_weights[k].view(parallel_shape+(1,)*(len(data_shape)-next_order)+(2,))
                expanded_out_weights = expanded_out_weights.expand_as(temp_tensor)
                temp_tensor = CUDAcpl.einsum('...,...->...',temp_tensor,expanded_out_weights)
            #broadcast according to the index distance
            temp_shape = temp_tensor.shape
            temp_tensor = temp_tensor.view(
                parallel_shape+(next_order-current_order-1)*(1,)+temp_shape[len(parallel_shape):])
            temp_tensor = temp_tensor.expand(
                parallel_shape+data_shape[current_order+1:next_order]+temp_shape[len(parallel_shape):])
            par_tensor.append(temp_tensor)
        
        return torch.stack(par_tensor,dim=len(parallel_shape))
        



    def layout(self, index_order: List[int], dot=Digraph(), succ: List=[], real_label: bool=True, full_output: bool=False):
        '''
            full_output: if True, then the edge will appear as a tensor, not the parallel index shape.

            (NO TYPING SYSTEM VERIFICATION)
        '''


        col=['red','blue','black','green']

        if real_label:
            if self.id==TERMINAL_KEY:
                dot.node(str(self.id), str(1), fontname="helvetica",shape="circle",color="red")
            else:
                dot.node(str(self.id), 'i'+str(index_order[self.order]), fontname="helvetica",shape="circle",color="red")
        else:
            dot.node(str(self.id), 'i'+str(index_order[self.order]), fontname="helvetica",shape="circle",color="red")

        for k in range(self.index_range):
            if self.successors[k]:

                #if there is no parallel index, directly demonstrate the edge values
                if list(self.out_weights[0].shape) == [2]:
                    label1=str(complex(round(self.out_weights[k][0].cpu().item(),2),round(self.out_weights[k][1].cpu().item().imag,2)))
                #otherwise, demonstrate the parallel index shape
                else:
                    if full_output:
                        label1 = str(CUDAcpl2np(self.out_weights[k]))
                    else:
                        label1 = str(list(self.out_weights[k].shape[:-1]))
                if not self.successors[k] in succ:
                    dot=self.successors[k].layout(index_order, dot,succ,real_label,full_output)
                    dot.edge(str(self.id),str(self.successors[k].id),color=col[k%4],label=label1)
                    succ.append(self.successors[k])
                else:
                    dot.edge(str(self.id),str(self.successors[k].id),color=col[k%4],label=label1)
        return dot        
