from __future__ import annotations
from typing import Tuple, Union, List, Dict

from . import CUDAcpl
from .CUDAcpl import CUDAcpl_Tensor, _U_
from .node import Node

import torch

'''
This source contains all the methods at the weighted node level.
'''

def isequal(node1: Node|None, weights1: CUDAcpl_Tensor, 
            node2: Node|None ,weights2: CUDAcpl_Tensor) -> bool:
    if node1 == node2 \
        and (Node.get_int_key(weights1)==Node.get_int_key(weights2)).all():
        return True
    else:
        return False
        

def normalized(node: Node|None, dangle_weights: CUDAcpl_Tensor, iterate: bool) -> Tuple[Node|None, CUDAcpl_Tensor]:
    '''
        Conduct the normalization of this node.
        Return the normalized node and normalization coefficients.

        If iterate is True, then the normalization will be conducted from top to bottom.
        Otherwise, it is only conducted for this node,
        and assume its successors are all normalized already.
    '''

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
                node_normalized, out_weight = normalized(succ, node.out_weights[k],True)
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
        if not isequal(node.successors[k], weigs[k], node.successors[0], weigs[0]):
            all_equal = False
            break
    if all_equal:
        new_dangle_weights = CUDAcpl.einsum('...,...->...',dangle_weights,weigs[0])
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
    new_dangle_weights = CUDAcpl.einsum('...,...->...',dangle_weights,weig_max)
    return  new_node, new_dangle_weights


def to_CUDAcpl_Tensor(node: Node|None, weights: CUDAcpl_Tensor, data_shape: List[int]) -> CUDAcpl_Tensor:
    '''
        Get the CUDAcpl_Tensor determined from this node and the weights.

        (use the trival index order)
        data_shape(in the corresponding trival index order) is required,
            for the result should broadcast at reduced nodes of indices.
    '''

    parallel_shape = tuple(weights.shape[:-1])

    if node == None:
        res = CUDAcpl.ones(parallel_shape)
        n_extra_one = 0
    else:
        tensor_dict = dict()
        res = __to_CUDAcpl_Tensor(node, weights, data_shape, tensor_dict)
        n_extra_one = node.order
    
    #this extra layer is for adding the reduced dimensions at the front
    res = res.view(parallel_shape+n_extra_one*(1,)+res.shape[len(parallel_shape):])
    res = res.expand(parallel_shape + tuple(data_shape)+(2,))

    return res
    


def __to_CUDAcpl_Tensor(node: Node, weights: CUDAcpl_Tensor, data_shape: List[int], tensor_dict: Dict) -> CUDAcpl_Tensor:
    
    current_order = node.order

    parallel_shape = tuple(weights.shape[:-1])
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
                temp_tensor = tensor_dict[key]
                next_order = succ.order
            else:
                temp_tensor = __to_CUDAcpl_Tensor(succ, node.out_weights[k],data_shape, tensor_dict)
                next_order = succ.order
                expanded_out_weights = node.out_weights[k].view(parallel_shape+(1,)*(len(data_shape)-next_order)+(2,))
                expanded_out_weights = expanded_out_weights.expand_as(temp_tensor)
                temp_tensor = CUDAcpl.einsum('...,...->...',temp_tensor,expanded_out_weights)
                #add into the dictionary
                tensor_dict[key] = temp_tensor
        #broadcast according to the index distance
        temp_shape = temp_tensor.shape
        temp_tensor = temp_tensor.view(
            parallel_shape+(next_order-current_order-1)*(1,)+temp_shape[len(parallel_shape):])
        temp_tensor = temp_tensor.expand(
            parallel_shape+tuple(data_shape[current_order+1:next_order])+temp_shape[len(parallel_shape):])
        par_tensor.append(temp_tensor)
    
    return torch.stack(par_tensor,dim=len(parallel_shape))
    
