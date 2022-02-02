from __future__ import annotations
from enum import unique
from typing import Tuple, Union, List, Dict, cast


from . import CUDAcpl
from .CUDAcpl import CUDAcpl_Tensor, _U_
from .node import Node

import torch
from graphviz import Digraph
from IPython.display import Image

'''
This source contains all the methods at the weighted node level.
'''

WeightedNode = Tuple[Union[Node,None], CUDAcpl_Tensor]

def isequal(w_node1: WeightedNode, w_node2: WeightedNode) -> bool:
    if w_node1[0] == w_node2[0] \
        and (Node.get_int_key(w_node1[1])==Node.get_int_key(w_node2[1])).all():
        return True
    else:
        return False
        

def normalize(w_node: WeightedNode, iterate: bool) -> WeightedNode:
    '''
        Conduct the normalization of this node.
        Return the normalized node and normalization coefficients.

        If iterate is True, then the normalization will be conducted from top to bottom.
        Otherwise, it is only conducted for this node,
        and assume its successors are all normalized already.
    '''
    node, dangle_weights = w_node

    if node == None:
        return None, dangle_weights

    #redirect all_zero edges to the terminal node
    int_key = Node.get_int_key(dangle_weights)
    if torch.max(int_key) == 0 and torch.min(int_key) == 0:
        new_dangle_weights = _U_(torch.zeros_like,dangle_weights)
        return None, new_dangle_weights

    # normalize the successors first if iterate is True
    if iterate:
        out_weights = []
        out_nodes = []
        for k in range(len(node.successors)):
            succ = node.successors[k]
            if succ == None:
                node_normalized = None
                out_weight = node.out_weights[k]
            else:
                node_normalized, out_weight = normalize((succ, node.out_weights[k]),True)
            out_nodes.append(node_normalized)
            out_weights.append(out_weight)
        #pay attention that out_weigs are stacked at the first index here
        weigs = torch.stack(out_weights)
    else:
        out_nodes = node.successors
        weigs = node.out_weights

    #node reduction check (reduce when all equal)
    all_equal = True
    for k in range(1,node.index_range):
        if not isequal((node.successors[k], weigs[k]), (node.successors[0], weigs[0])):
            all_equal = False
            break
    if all_equal:
        new_dangle_weights = CUDAcpl.mul_element_wise(dangle_weights,weigs[0])
        return out_nodes[0], new_dangle_weights


    #this zero-checking is not written in CUDA because we expect it to ternimate very soon
    all_zeros = True
    for k in range(node.index_range):
        int_key = Node.get_int_key(weigs[k])
        if torch.max(int_key) != 0 or torch.min(int_key) != 0:
            all_zeros=False
            break
    if all_zeros:
        weights=_U_(torch.zeros_like,dangle_weights)
        return None, weights

    #get the maximum abs. of weights
    weig_abs = CUDAcpl.norm(weigs)
    weig_abs_max = torch.max(weig_abs,dim=0,keepdim=True)
    max_indices = weig_abs_max.indices
    max_abs_values = weig_abs_max.values[0]

    #get the weights at the corresponding maximum abs. positions
    weig_max = torch.stack((weigs[...,0].gather(0,max_indices),weigs[...,1].gather(0,max_indices)),dim=-1).squeeze(0) #shape: [?,2]

    #get 1/weight_max for normalization
    weig_max_1 = torch.stack((weig_max[...,0],-weig_max[...,1]),dim=-1)/(weig_max[...,0]**2+weig_max[...,1]**2).unsqueeze(-1)

    #notice that weig_max == 0 cases are considered here
    zeros_items = (max_abs_values<Node.EPS).unsqueeze(-1).broadcast_to(weig_max_1.shape)
    weig_max_1 = torch.where(zeros_items,0.,weig_max_1)

    weigs = CUDAcpl.einsum('k...,...->k...',weigs,weig_max_1)   #shape: [weigs_index,?,2],[?,2]->[weigs_index,?,2]

    new_node=Node.get_unique_node(node.order,weigs,out_nodes)

    #calculate the new dangling weight
    new_dangle_weights = CUDAcpl.mul_element_wise(dangle_weights,weig_max)
    return  new_node, new_dangle_weights


def to_CUDAcpl_Tensor(w_node: WeightedNode, data_shape: List[int]) -> CUDAcpl_Tensor:
    '''
        Get the CUDAcpl_Tensor determined from this node and the weights.

        (use the trival index order)
        data_shape(in the corresponding trival index order) is required,
            for the result should broadcast at reduced nodes of indices.
    '''
    node, weights = w_node
    parallel_shape = tuple(weights.shape[:-1])

    if node == None:
        res = CUDAcpl.ones(parallel_shape)
        n_extra_one = 0
    else:
        tensor_dict = dict()
        res = __to_CUDAcpl_Tensor(node, data_shape, tensor_dict)
        n_extra_one = node.order
    
    #this extra layer is for adding the reduced dimensions at the front
    res = res.view(parallel_shape+n_extra_one*(1,)+res.shape[len(parallel_shape):])
    res = res.expand(parallel_shape + tuple(data_shape)+(2,))

    #multiply by the dangling weights
    weigths_broadcasted = weights.view(weights.shape[:-1]+(1,)*len(data_shape)+(2,)).expand_as(res)
    res = CUDAcpl.mul_element_wise(weigths_broadcasted, res)
    return res
    


def __to_CUDAcpl_Tensor(node: Node, data_shape: List[int], tensor_dict: Dict) -> CUDAcpl_Tensor:
    '''
        tensor_dict: caches the corresponding tensor of this node (weights = 1)
    '''

    current_order = node.order

    parallel_shape = node.out_weights.shape[1:-1]
    par_tensor = []
    for k in range(node.index_range):
        #detect terminal nodes, or iterate on the next node
        succ = node.successors[k]
        if succ == None:
            temp_tensor = node.out_weights[k]
            next_order = len(data_shape)    
        else:
            #first look up in the dictionary 
            key = succ.unique_key
            if key in tensor_dict:
                uniform_tensor = tensor_dict[key]
            else:
                uniform_tensor = __to_CUDAcpl_Tensor(succ, data_shape, tensor_dict)
                #add into the dictionary
                tensor_dict[key] = uniform_tensor
            next_order = succ.order
            expanded_out_weights = node.out_weights[k].view(parallel_shape+(1,)*(len(data_shape)-next_order)+(2,))
            expanded_out_weights = expanded_out_weights.expand_as(uniform_tensor)
            temp_tensor = CUDAcpl.mul_element_wise(uniform_tensor,expanded_out_weights)
        #broadcast according to the index distance
        temp_shape = temp_tensor.shape
        temp_tensor = temp_tensor.view(
            parallel_shape+(next_order-current_order-1)*(1,)+temp_shape[len(parallel_shape):])
        temp_tensor = temp_tensor.expand(
            parallel_shape+tuple(data_shape[current_order+1:next_order])+temp_shape[len(parallel_shape):])
        par_tensor.append(temp_tensor)
    
    res = torch.stack(par_tensor,dim=len(parallel_shape))
    return res
    

def index_single(w_node: WeightedNode, inner_index: int, key: int) -> WeightedNode:
    '''
        Indexing on the single index. Again, inner_index indicate that of tdd nodes DIRECTLY.

        Detail: normalization is conducted level wise.
    '''
    indexed_dict = dict()
    return __index_single(indexed_dict, w_node, inner_index, key)

def __index_single(indexed_dict: Dict, w_node: WeightedNode, inner_index: int, key: int) -> WeightedNode:
    '''
    indexed_dict are used for caching the already indexed weighted nodes.
    '''
    node, dangle_weights = w_node

    if node == None:
        return None, dangle_weights
    
    unique_key = node.unique_key

    #first look up the indexing result in the dict
    if unique_key in indexed_dict:
        # Note (VITAL) : here res_weights_below represents the corresponding indexing result of node,
        #  assuming dangling weights = 1.
        res_node, res_weights_below = indexed_dict[unique_key]
            
    else:
        if inner_index < node.order:
            res_node = Node.shift(node, -1)
            res_weights_below = CUDAcpl.ones(dangle_weights.shape[:-1])
        elif inner_index == node.order:
            #note that order should decrese 1 due to indexing
            res_node = node.successors[key]
            if res_node != None:
                res_node = Node.shift(res_node, -1)
            res_weights_below = node.out_weights[key]
        else:
            out_nodes = []
            out_weights = []
            for k in range(node.index_range):
                succ = node.successors[k]
                if succ == None:
                    out_nodes.append(None)
                    out_weights.append(node.out_weights[k])
                else:
                    temp_node, temp_weights = __index_single(indexed_dict,(succ,node.out_weights[k]),inner_index,key)
                    out_nodes.append(temp_node)
                    out_weights.append(temp_weights)

            new_weights = torch.stack(out_weights)
            new_node = Node(0,node.order,new_weights,out_nodes)
            res_node, res_weights_below =  normalize((new_node, 
                                        CUDAcpl.ones(dangle_weights.shape[:-1])), False)
        #append to the dict cache
        indexed_dict[unique_key] = res_node, res_weights_below

    #multiply at the end
    new_dangle_weights = CUDAcpl.mul_element_wise(dangle_weights, res_weights_below)
    return res_node, new_dangle_weights

def index(w_node: WeightedNode, inner_indices: List[Tuple[int,int]]) -> WeightedNode:
    '''
        Return the indexed tdd according to the chosen keys at given indices.

        Note that here inner_indices indicates that of tdd nodes DIRECTLY.

        indices: [(index1, key1), (index2, key2), ...]
    '''
    indexing = list(inner_indices).copy()
    indexing.sort(key=lambda item: item[0])

    res_node, res_weights = w_node

    while indexing != []:
        res_node, res_weights = index_single((res_node, res_weights), indexing[0][0], indexing[0][1])
        #the indices are guaranteed to be in order
        indexing = [(indexing[i+1][0]-1,indexing[i+1][1]) for i in range(len(indexing)-1)]
    
    return res_node, res_weights

def sum(w_node1: WeightedNode, w_node2: WeightedNode) -> WeightedNode:
    '''
        Sum up the given weighted nodes, and return the reduced weighted node result.
    '''
    # (dictionary cache technique seems impossible for summation)

    node1, dangle_weights1 = w_node1
    node2, dangle_weights2 = w_node2
    if node1 == None and node2 == None:
        return None, dangle_weights1 + dangle_weights2

    out_nodes = []
    out_weights = []
    
    if node1 != None and node2 != None and node1.order == node2.order:
        for i in range(node1.index_range):
            next_weights1 = CUDAcpl.mul_element_wise(dangle_weights1, node1.out_weights[i])
            next_weights2 = CUDAcpl.mul_element_wise(dangle_weights2, node2.out_weights[i])
            temp_node, temp_weights = sum((node1.successors[i], next_weights1),
                                            (node2.successors[i], next_weights2))
            out_nodes.append(temp_node)
            out_weights.append(temp_weights)
        
        A = w_node1

    else: 
        '''
            There are three cases following, corresponding to the same procedure:
            1. node1 == None, node2 != None
            2. node2 == None, node1 != None
            3. node1 != None, node2 != None, but node1.order != noder2.order
            We first analysis the situation to reuse the codes.
            A will be the lower ordered weighted node.
        '''
        if node1 == None:
            A, B = w_node2, w_node1
        elif node2 == None:
            A, B = w_node1, w_node2
        else:
            if node1.order < node2.order:
                A, B = w_node1, w_node2
            else:
                A, B = w_node2, w_node1

        for i in range(A[0].index_range): # type: ignore
            next_weights_A = CUDAcpl.mul_element_wise(A[1], A[0].out_weights[i]) # type: ignore
            temp_node, temp_weights = sum((A[0].successors[i], next_weights_A), B) # type: ignore
            out_nodes.append(temp_node)
            out_weights.append(temp_weights)

    temp_new_node = Node(0, A[0].order, torch.stack(out_weights), out_nodes)       # type: ignore
    return normalize((temp_new_node, CUDAcpl.ones(dangle_weights1.shape[:-1])), False)


