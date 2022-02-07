from __future__ import annotations
from enum import unique
from math import remainder
from typing import Sequence, Tuple, Union, List, Dict, cast


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


def to_CUDAcpl_Tensor(w_node: WeightedNode, data_shape: Tuple[int]) -> CUDAcpl_Tensor:
    '''
        Get the CUDAcpl_Tensor determined from this node and the weights.

        (use the trival index order)
        data_shape(in the corresponding inner index order) is required,
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
    


def __to_CUDAcpl_Tensor(node: Node, data_shape: Tuple[int], tensor_dict: Dict) -> CUDAcpl_Tensor:
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

def index(w_node: WeightedNode, inner_indices: Sequence[Tuple[int,int]]) -> WeightedNode:
    '''
        Return the indexed tdd according to the chosen keys at given indices (clone guaranteed).

        Note that here inner_indices indicates that of tdd nodes DIRECTLY.

        weights in w_node will not be modified.

        indices: [(index1, key1), (index2, key2), ...]
    '''
    indexing = list(inner_indices).copy()
    indexing.sort(key=lambda item: item[0])

    res_node, res_weights = w_node

    if indexing == []:
        res_weights = res_weights.clone()

    while indexing != []:
        res_node, res_weights = index_single((res_node, res_weights), indexing[0][0], indexing[0][1])
        #the indices are guaranteed to be in order
        indexing = [(indexing[i+1][0]-1,indexing[i+1][1]) for i in range(len(indexing)-1)]
    
    return res_node, res_weights


def __sum_weights_normalize(dangle_weights1: CUDAcpl_Tensor, dangle_weights2: CUDAcpl_Tensor)\
            -> Tuple[CUDAcpl_Tensor, CUDAcpl_Tensor, CUDAcpl_Tensor]:
    '''
        Process the weights, and normalize them as a whole. Return (new_weights1, new_weights2, renorm_coef).

        The strategy to produce the weights (for every individual element):
            maximum_norm := max(weight1.norm, weight2.norm)
            1. if maximum_norm > EPS: renormalize max_weight to 1.
            3. else key is 0000...,0000...
    '''

    # chose the larger norm party
    norm1 = CUDAcpl.norm(dangle_weights1)
    norm2 = CUDAcpl.norm(dangle_weights2)
    chose1_CUDAcpl = (norm1 > norm2).unsqueeze(-1).broadcast_to(dangle_weights1.shape)
    maximum_element = torch.where(chose1_CUDAcpl, dangle_weights1, dangle_weights2)
    maximum_norm = torch.maximum(norm1, norm2)

    zero_item_CUDAcpl = (maximum_norm < Node.EPS).unsqueeze(-1).broadcast_to(dangle_weights1.shape)

    # renormalization coefficient for zero items are left to be 1.
    renorm_coef = torch.where(zero_item_CUDAcpl, CUDAcpl.ones(dangle_weights1.shape[:-1]), maximum_element)

    # perform the renormalization
    new_dangle_weights1 = CUDAcpl.div_element_wise(dangle_weights1, renorm_coef)
    new_dangle_weights2 = CUDAcpl.div_element_wise(dangle_weights2, renorm_coef)
    return new_dangle_weights1, new_dangle_weights2, renorm_coef


def __sum(dict_cache: Dict, w_node1: WeightedNode, w_node2: WeightedNode, renorm_coef) -> WeightedNode:
    '''
        Sum up the given weighted nodes, multply the renorm_coef, and return the reduced weighted node result.
        Note that weights1 and weights2 as a whole should have been normalized,
         and renorm_coef is the coefficient.
    '''

    node1, dangle_weights1 = w_node1
    node2, dangle_weights2 = w_node2
    if node1 == None and node2 == None:
        return None, CUDAcpl.mul_element_wise(dangle_weights1 + dangle_weights2, renorm_coef)


    #produce the unique key and look up in the dictionary
    key_part1 = Node.get_unique_key_all(node1) + tuple(Node.get_int_key(dangle_weights1).view(-1).tolist())
    key_part2 = Node.get_unique_key_all(node2) + tuple(Node.get_int_key(dangle_weights2).view(-1).tolist())
    key1 = key_part1+key_part2
    #Note: the swapped key will also be stored, so here we do not need the other swapped one.
    if key1 in dict_cache:
        final_node, final_dangle_weights = dict_cache[key1]
        final_dangle_weights = CUDAcpl.mul_element_wise(renorm_coef, final_dangle_weights)
        return final_node, final_dangle_weights

    else:
        out_nodes = []
        out_weights = []
        
        if node1 != None and node2 != None and node1.order == node2.order:
            for i in range(node1.index_range):
                next_weights1 = CUDAcpl.mul_element_wise(dangle_weights1, node1.out_weights[i])
                next_weights2 = CUDAcpl.mul_element_wise(dangle_weights2, node2.out_weights[i])
                # normalize as a whole
                new_weights1, new_weights2, new_renorm_coef = __sum_weights_normalize(next_weights1, next_weights2)
                temp_node, temp_weights = __sum(dict_cache, (node1.successors[i], new_weights1),
                                                (node2.successors[i], new_weights2), new_renorm_coef)
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
                # normalize as a whole
                new_weights_A, new_weights_B, new_renorm_coef = __sum_weights_normalize(next_weights_A, B[1])
                temp_node, temp_weights = __sum(dict_cache, (A[0].successors[i], new_weights_A), (B[0], new_weights_B), new_renorm_coef) # type: ignore
                out_nodes.append(temp_node)
                out_weights.append(temp_weights)
        
        temp_new_node = Node(0, A[0].order, torch.stack(out_weights), out_nodes)       # type: ignore
        final_node, final_weights = normalize((temp_new_node, CUDAcpl.ones(dangle_weights1.shape[:-1])), False)

        key2 = key_part2 + key_part1
        dict_cache[key1] = final_node, final_weights
        dict_cache[key2] = final_node, final_weights
        '''
            In fact, here we can check whether the two weights are opposite numbers, and add to the dictionary.
            Because when dangle_weights1 + dangle_weights2 == 0, they will be normalized randomly.
            But for parallel case it is too hard to implement.
        '''
        #finally multiply the renorm_coef
        final_weights = CUDAcpl.mul_element_wise(final_weights, renorm_coef)
        return final_node, final_weights


def sum(w_node1: WeightedNode, w_node2: WeightedNode, dict_cache: Dict= None) -> WeightedNode:
    if dict_cache == None:
        dict_cache = dict()
    new_weights1, new_weights2, new_reform_coef = __sum_weights_normalize(w_node1[1], w_node2[1])
    return __sum(dict_cache, (w_node1[0], new_weights1), (w_node2[0], new_weights2), new_reform_coef)


def __contract(dict_cache: Dict, w_node: WeightedNode, data_shape: Tuple[int,...], 
            remained_ils: Tuple[Tuple[int,...]|Tuple[()],Tuple[int,...]|Tuple[()]], 
            waiting_ils: Tuple[Tuple[int,...]|Tuple[()],Tuple[int,...]|Tuple[()]],
            sum_dict_cache: Dict= None) -> WeightedNode:
    '''
        dict_cache: is used to cache the calculated weighted node (cached results assume the dangling weight to be 1)
        sum_dict_cache: the dict_cache from former calculations

        remained_ils: the indices not processed yet
            (which starts to trace, and is waiting for the second index) 
        waiting_ils: the list of indices waiting to be traced. 
            format: ((waiting_index1, waiting_index2, ...), (i_val1, i_val2, ...))

        Note that:
            1. For remained_ils, we require smaller indices to be in the first list, 
                and the corresponding larger one in the second.
            2. waiting_ils[0] must be sorted in the asscending order to keep the dict key unique.
            3. The returning weighted nodes are NOT shifted. (node.order not adjusted)
    '''
    node, dangle_weights = w_node

    if node == None:
        #close all the unprocessed indices
        scale = 1.
        for i in remained_ils[0]:
            scale *= data_shape[i]
        new_dangle_weights = dangle_weights*scale
        return None, new_dangle_weights
    
    #first look up in the dictionary
    key = node.unique_key + remained_ils + waiting_ils
    if key in dict_cache:
        final_node, final_weights_below = dict_cache[key]
        return final_node, CUDAcpl.mul_element_wise(dangle_weights, final_weights_below)
    else:
        #store the scaling number due to skipped remained indices
        scale = 1.

        #process the skipped remained indices (situations only first index skipped will not be processed afterwards)
        temp_remained_ils_0 = []
        temp_remained_ils_1 = []
        for i in range(len(remained_ils[0])):
            if remained_ils[1][i] >= node.order:
                temp_remained_ils_0.append(remained_ils[0][i])
                temp_remained_ils_1.append(remained_ils[1][i])
            else:
                scale *= data_shape[remained_ils[0][i]]
        remained_ils = (tuple(temp_remained_ils_0), tuple(temp_remained_ils_1))

        #process the skipped waiting indices
        temp_waiting_ils_i = []
        temp_waiting_ils_v = []
        for i in range(len(waiting_ils[0])):
            #if a waiting index is skipped, remove it from the waiting index list
            if waiting_ils[0][i] >= node.order:
                temp_waiting_ils_i.append(waiting_ils[0][i])
                temp_waiting_ils_v.append(waiting_ils[1][i])
        waiting_ils = (tuple(temp_waiting_ils_i), tuple(temp_waiting_ils_v))

        #the flag for no operation performed
        not_operated = True

        #check whether all operations have already taken place
        if len(remained_ils[0])==0 and len(waiting_ils[0]) == 0:
            final_node = node
            final_weights_below = CUDAcpl.ones(dangle_weights.shape[:-1]) * scale
            not_operated = False

        elif len(waiting_ils[0]) != 0:
            '''
            waiting_ils is not empty in this case
            If multiple waiting indices have been skipped, we will resolve with iteration, one by one.
            '''
            next_i_to_close = min(waiting_ils[0])
            #note that next_i_to_close >= node,order is guaranteed here
            if node.order == next_i_to_close:
                #close the waiting index
                ls_pos = waiting_ils[0].index(next_i_to_close)
                next_w_ils = (waiting_ils[0][:ls_pos]+waiting_ils[0][ls_pos+1:], waiting_ils[1][:ls_pos]+waiting_ils[1][ls_pos+1:])
                final_node, new_dangle_weights = __contract(dict_cache, 
                                    (node.successors[waiting_ils[1][ls_pos]], node.out_weights[waiting_ils[1][ls_pos]]),
                                    data_shape, remained_ils, next_w_ils, sum_dict_cache)
                final_weights_below = new_dangle_weights*scale
                not_operated = False

        if len(remained_ils[0]) != 0 and not_operated:
            '''
            Check the remained indices to start tracing.
            If multiple (smaller ones of) remained indices have been skipped, we will resolve with iteration, one by one.
            '''
            next_i_to_open = min(remained_ils[0])
            if node.order >= next_i_to_open:
                #open the index and finally sum up
                ls_pos = remained_ils[0].index(next_i_to_open)

                # next_r_ils: sorted.
                next_r_ils = ((remained_ils[0][:ls_pos]+remained_ils[0][ls_pos+1:]),
                                (remained_ils[1][:ls_pos]+remained_ils[1][ls_pos+1:]))
                #find the right insert place in waiting_ils
                pos = 0
                for index in waiting_ils[0]:
                    if index < next_i_to_open:
                        pos += 1
                    else:
                        break

                out_nodes = []
                out_weights = []
                if node.order == next_i_to_open:
                    for i in range(node.index_range):
                        succ = node.successors[i]
                        if succ == None:
                            new_node = None
                            new_weights = node.out_weights[i]
                        else:
                            ##################################
                            #produce the sorted new index lists
                            next_w_ils = (waiting_ils[0][:pos]+(remained_ils[1][ls_pos],)+waiting_ils[0][pos:],
                                             waiting_ils[1][:pos]+(i,)+waiting_ils[1][pos:])   

                            new_node, new_weights = __contract(dict_cache, (succ, node.out_weights[i]),data_shape,next_r_ils,next_w_ils, sum_dict_cache)
                        out_nodes.append(new_node)
                        out_weights.append(new_weights)
                else:
                    #this node skipped the index next_i_to_open in this case
                    for i in range(data_shape[next_i_to_open]):
                        ##################################
                        #produce the sorted new index lists
                        next_w_ils = (waiting_ils[0][:pos]+(remained_ils[1][ls_pos],)+waiting_ils[0][pos:],
                                            waiting_ils[1][:pos]+(i,)+waiting_ils[1][pos:])   

                        new_node, new_weights = __contract(dict_cache, w_node, data_shape, next_r_ils, next_w_ils, sum_dict_cache)
                        out_nodes.append(new_node)
                        out_weights.append(new_weights)

                #however the subnode outcomes are calculated, sum them over.
                final_node, res_weights = out_nodes[0], out_weights[0]
                for i in range(1,node.index_range):
                    final_node, res_weights = sum((final_node, res_weights), (out_nodes[i], out_weights[i]), sum_dict_cache)
                final_weights_below = res_weights*scale
                not_operated=False


        if not_operated:
            #in this case, no operation can be performed on this node, so we move on the the following nodes.
            out_nodes = []
            out_weights = []
            for i in range(node.index_range):
                succ = node.successors[i]
                if succ == None:
                    new_node = None
                    new_weights = node.out_weights[i]
                else:
                    new_node, new_weights = __contract(dict_cache, (succ, node.out_weights[i]),data_shape,remained_ils,waiting_ils, sum_dict_cache)
                out_nodes.append(new_node)
                out_weights.append(new_weights)
            final_node = Node(0, node.order, torch.stack(out_weights), out_nodes)
            final_node, final_weights_below = normalize((final_node, CUDAcpl.ones(dangle_weights.shape[:-1])*scale), False)    
    
    dict_cache[key] = final_node, final_weights_below # type: ignore
    return final_node, CUDAcpl.mul_element_wise(dangle_weights, final_weights_below) # type: ignore




        

def contract(w_node: WeightedNode, data_shape: Tuple[int,...],
         data_indices: Sequence[Sequence[int]],
         sum_dict_cache: Dict= None) -> WeightedNode:
    '''
        Trace the weighted node according to the specified data_indices. Return the reduced result.
        data_shape: correponds to data_indices
        data_indices should be counted in the data indices only.
        e.g. ([a,b,c],[d,e,f]) means tracing indices a-d, b-e, c-f (of course two lists should be in the same size)
        (smaller indices are required to be in the first list.)
    '''

    #at least the summation results should be cached during one contraction
    if sum_dict_cache == None:
        sum_dict_cache = dict()


    dict_cache = dict()
    res_node, res_weights = __contract(dict_cache, w_node, data_shape, 
                (tuple(data_indices[0]),tuple(data_indices[1])),((),()), sum_dict_cache)

    #shift the nodes at a time
    new_order_ls = list(range(len(data_shape)))
    reduced_indices = data_indices[0]+data_indices[1] # type: ignore
    reduced_indices = sorted(reduced_indices)
    for i in range(len(reduced_indices)-1):
        for j in range(reduced_indices[i]+1, reduced_indices[i+1]):
            new_order_ls[j] = j-i-1
    for j in range(reduced_indices[-1]+1, len(data_shape)):
        new_order_ls[j] = j-len(reduced_indices)


    return Node.shift_multiple(res_node, new_order_ls), res_weights
    

