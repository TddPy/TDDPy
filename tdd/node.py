from __future__ import annotations
from typing import List, Tuple, Union, cast
from .CUDAcpl import CUDAcpl_Tensor,_U_
import torch

from graphviz import Digraph

TERMINAL_KEY = 0

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
            [index(weight1,weight2...),id1,id2,...] for non-terminal nodes
    '''
    global_node_id = 0 #it counts the total number of nonterminal nodes

    @staticmethod
    def reset():
        Node.__unique_table.clear()
        Node.global_node_id = 0
        
        terminal_node = Node()
        terminal_node.id = 0
        terminal_node.out_weights = _U_(torch.tensor,[])
        terminal_node.successors = []

        Node.__unique_table[TERMINAL_KEY] = terminal_node


    def __init__(self):
        '''
        The structure of node instances:
        - id
        - out_weights : torch.Tensor, shape: [succ_num, ..., 2]. The first index is for the successors,
                        and the last index is for complex representation.
        - successor
        '''
        self.id : int = 0
        self.out_weights : CUDAcpl_Tensor = _U_(torch.tensor,[])
        self.successors : List[Node] = []

    @staticmethod
    def get_terminal_node():
        return Node.__unique_table[TERMINAL_KEY]

    @staticmethod
    def get_unique_node(out_weights: CUDAcpl_Tensor, succ_nodes: List[Node]) -> Node:
        '''
            Return the required node. It is either from the unique table, or a newly created one.
            
            id: the node id, or -1 to directly obtain the terminal node
            out_weights: the incoming weights of this node, shape: [succ_num, ..., 2].
            succ_nodes: the successor nodes.

            Note: The equality checking inside is conducted with the node.EPS tolerance. So feel free
                    to pass in the raw weights from calculation.
        '''
                
        #generate the unique key
        temp_key : Tuple[int|Node]
        temp_key = tuple(cast(List[Union[int,Node]],Node.get_int_key(out_weights).view(-1).tolist()) 
                    + cast(List[Union[int,Node]],succ_nodes))

        if temp_key in Node.__unique_table:
            return Node.__unique_table[temp_key]
        else:
            res = Node()
            Node.global_node_id += 1
            res.id = Node.global_node_id
            res.out_weights = out_weights.clone().detach()
            res.successors = succ_nodes.copy()
            Node.__unique_table[temp_key] = res
            return res



    def layout(self,index_order,dot=Digraph(),succ=[],real_label=True):
        col=['red','blue','black','green']

        if real_label:
            if self.id==TERMINAL_KEY:
                dot.node(str(self.id), str(1), fontname="helvetica",shape="circle",color="red")
            else:
                dot.node(str(self.id), 'i'+str(index_order[0]), fontname="helvetica",shape="circle",color="red")
        else:
            dot.node(str(self.id), 'i'+str(index_order[0]), fontname="helvetica",shape="circle",color="red")

        for k in range(len(self.successors)):
            if self.successors[k]:

                #if there is no parallel index, directly demonstrate the edge values
                if list(self.out_weights[0].shape) == [2]:
                    label1=str(complex(round(self.out_weights[k][0].cpu().item(),2),round(self.out_weights[k][1].cpu().item().imag,2)))
                #otherwise, demonstrate the parallel index shape
                else:
                    label1 = str(list(self.out_weights[0].shape[:-1]))
                if not self.successors[k] in succ:
                    dot=self.successors[k].layout(index_order[1:],dot,succ,real_label)
                    dot.edge(str(self.id),str(self.successors[k].id),color=col[k%4],label=label1)
                    succ.append(self.successors[k])
                else:
                    dot.edge(str(self.id),str(self.successors[k].id),color=col[k%4],label=label1)
        return dot        
