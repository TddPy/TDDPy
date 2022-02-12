from __future__ import annotations
from typing import Dict, Iterable, Sequence, Tuple, List, Any, Union, cast
import numpy as np
import torch

from . import CUDAcpl, weighted_node
from .CUDAcpl import _U_, CUDAcpl_Tensor, CUDAcpl2np
from .node import  TERMINAL_ID, Node, order_inverse
from .weighted_node import isequal, to_CUDAcpl_Tensor


from graphviz import Digraph
from IPython.display import Image



class TDD:
    '''
        TDD  functions as the compact representation of tensors,
        and can fit into tensor networks.
    '''
    def __init__(self, 
                    weights: CUDAcpl_Tensor,
                    data_shape: Sequence[int],
                    node: Node|None,
                    index_order: Sequence[int] = []):
        self.__weights: CUDAcpl_Tensor = weights
        self.__data_shape: List[int] = list(data_shape)  #the data index shape (of the tensor it represents)
        self.__node: Node|None = node

        '''
            index_order: how the inner index are mapped to outer representations
            for example, tdd[a,b,c] under index_order=[0,2,1] returns the value tdd[a,c,b]
            index_order == None means the trival index mapping [0,1,2,(...)]
        '''
        self.__index_order: List[int] = list(index_order)

    @property
    def node(self) -> Node|None:
        return self.__node

    @property
    def weights(self) -> CUDAcpl_Tensor:
        return self.__weights

    @property
    def data_shape(self) -> Tuple[int,...]:
        return tuple(self.__data_shape)

    @property
    def index_order(self) -> Tuple[int,...]:
        return tuple(self.__index_order)

    @property
    def dim_data(self) -> int:
        return len(self.__data_shape)

    @property
    def parallel_shape(self) -> Tuple[int,...]:
        return tuple(self.__weights.shape[:-1])

    @property
    def global_order(self)-> Tuple[int,...]:
        '''
            Return the index order containing both parallel and data indices.
            Note that the last index reserved for CUDA complex is not included
        '''
        parallel_index_order = [i for i in range(len(self.parallel_shape))]
        increment = len(self.parallel_shape)
        return tuple(parallel_index_order + [order+increment for order in self.__index_order])
    
    @property
    def global_shape(self)-> Tuple[int,...]:
        return tuple(self.parallel_shape + self.data_shape)

    def __eq__(self, other: TDD) -> bool:
        '''
            Now this equality check only deals with TDDs with the same index order.
        '''
        res = self.__index_order==other.__index_order \
            and isequal((self.__node,self.__weights),(other.__node,other.__weights))
        return res

    def __str__(self):
        return str(self.numpy())

    def get_size(self) -> int:
        if self.__node == None:
            return 0
        return self.__node.get_size()

    @staticmethod
    def __as_tensor_iterate(tensor : CUDAcpl_Tensor, 
                    parallel_shape: List[int],
                    data_shape: List[int],
                    index_order: List[int], depth: int) -> TDD:
        '''
            The inner interation for as_tensor.

            tensor: will be referred to without cloning
            depth: current iteration depth, used to indicate index_order and termination
            index_order should not be []

            Guarantee: parallel_shape and index_order will not be modified.
        '''

        #checks whether the tensor is reduced to the [[...[val]...]] form
        if depth == len(data_shape):

            #maybe some improvement is needed here.
            if len(data_shape)==0:
                weights = tensor
            else:
                weights = (tensor[...,0:1,:]).view(parallel_shape+[2])
            res = TDD(weights,[],None,[])
            return res
        

        split_pos=index_order[depth]
        split_tensor = list(tensor.split(1,-len(data_shape)+split_pos-1))
            #-1 is because the extra inner dim for real and imag

        the_successors: List[TDD] =[]

        for k in range(data_shape[split_pos]):
            res = TDD.__as_tensor_iterate(split_tensor[k],parallel_shape,data_shape,index_order,depth+1)
            the_successors.append(res)

        #stack the sub-tdd
        succ_nodes = [item.__node for item in the_successors]
        out_weights = torch.stack([item.__weights for item in the_successors])
        temp_node = Node(0, depth, out_weights, succ_nodes)
        dangle_weights = CUDAcpl.ones(out_weights.shape[1:-1])
        #normalize at this depth
        new_node, new_dangle_weights = weighted_node.normalize((temp_node, dangle_weights), False)
        tdd = TDD(new_dangle_weights, [], new_node, [])

        return tdd


    @staticmethod
    def as_tensor(data : TDD|CUDAcpl_Tensor|np.ndarray|
        Tuple[CUDAcpl_Tensor|np.ndarray, Sequence[int], Sequence[int]]) -> TDD:
        '''
        construct the tdd tensor

        tensor:
            0. in the form of a TDD, then return a copy of it.
            1. in the form of a matrix only: assume the parallel index and index order to be []
            2. in the form of a tuple (data, index_shape, index_order)
            Note that if the input matrix is a torch tensor, 
                    then it must be already in CUDAcpl_Tensor(CUDA complex) form.
        '''
        if isinstance(data,TDD):
            return data.clone()

        if isinstance(data,Tuple):
            tensor,parallel_shape,index_order = data
        else:
            tensor = data
            parallel_shape = []
            index_order = []
            
        if isinstance(tensor,np.ndarray):
            tensor = CUDAcpl.np2CUDAcpl(tensor)

        #pre-process above

        data_shape = list(tensor.shape[len(parallel_shape):-1])  #the data index shape
        index_order = list(index_order)
        if index_order == []:
            result_index_order = list(range(len(data_shape)))
        else:
            result_index_order = index_order.copy()


        if len(data_shape)!=len(result_index_order):
            raise Exception('The number of indices must match that provided by tensor.')

        '''
            This extra layer is also for copying the input list and pre-process.
        '''
        res = TDD.__as_tensor_iterate(tensor,list(parallel_shape),data_shape,result_index_order,0)

        
        res.__index_order = result_index_order
        res.__data_shape = data_shape
        return res

    @property
    def __inner_data_shape(self) -> Tuple[int,...]:
        '''
            return the corresponding inner data shape due to the index_order
        '''
        dim = self.dim_data
        res = [0]*dim
        for i in range(dim):
            res[i] = self.__data_shape[self.__index_order[i]]
        return tuple(res)
    
            
    def CUDAcpl(self) -> CUDAcpl_Tensor:
        '''
            Transform this tensor to a CUDA complex and return.
        '''
        node_data = to_CUDAcpl_Tensor((self.__node,self.__weights),self.__inner_data_shape)
        
        #permute to the right index order
        node_data = node_data.permute(self.global_order+(node_data.dim()-1,))

        return node_data
        

    def numpy(self) -> np.ndarray:
        '''
            Transform this tensor to a numpy ndarry and return.
        '''
        return CUDAcpl2np(self.CUDAcpl())


    def clone(self) -> TDD:
        return TDD(self.__weights.clone(), self.__data_shape.copy(), self.__node, self.__index_order.copy())

    def get_view(self) -> TDD:
        '''
            Return a view of this tdd.
        '''
        return TDD(self.__weights, self.__data_shape.copy(), self.__node, self.__index_order.copy())

        '''
    
    def __getitem__(self, key) -> TDD:
        Index on the data dimensions.

        Note that only limited form of indexing is allowed here.
        if not isinstance(key, int):
            raise Exception('Indexing form not supported.')
        
        # index by a integer
        inner_index = self.index_order.index(key) #get the corresponding index inside tdd
        node = self.node.
        '''
    

    def __index_reduce_proc(self, reduced_indices: Sequence[int])-> Tuple[List[int], List[int]]:
        '''
            Return the data_shape and index_order after the reduction of specified indices.
            reduced_indices: corresponds to inner data indices, not the indices of tensor it represents.
            Note: Indices are counted in data indices only.
        '''
        raise Exception("bug here. innershape should be used. data shape should also be sorted.")
        new_data_shape = []
        indexed_index_order = []
        for i in range(len(self.__data_shape)):
            if i not in reduced_indices:
                new_data_shape.append(self.__data_shape[i])
                indexed_index_order.append(self.__index_order[i])        
        new_index_order = sorted(range(len(indexed_index_order)), key = lambda k:indexed_index_order[k])

        return new_data_shape, new_index_order

    
    def index(self, data_indices: Sequence[Tuple[int,int]]) -> TDD:
        '''
        Return the indexed tdd according to the chosen keys at given indices.
        Note: Indices should be count in the data indices only.

        Note: indexing acts on the data indices.

        indices: [(index1, key1), (index2, key2), ...]
        '''
        #transform to inner indices
        reversed_order = order_inverse(self.__index_order)
        inner_indices = [(reversed_order[item[0]],item[1]) for item in data_indices]

        #get the indexing of inner data
        new_node, new_dangle_weights = weighted_node.index((self.__node, self.__weights), inner_indices)
        
        #process the data_shape and the index_order
        indexed_indices = []
        for pair in inner_indices:
            indexed_indices.append(pair[0])
        new_data_shape, new_index_order = self.__index_reduce_proc(indexed_indices)

        return TDD(new_dangle_weights, new_data_shape, new_node, new_index_order)

    @staticmethod
    def sum(a: TDD, b: TDD) -> TDD:
        '''
            Sum up tdd a and b, and return the reduced result. 
        '''

        new_node, new_weights = weighted_node.sum((a.__node, a.__weights), (b.__node, b.__weights))

        return TDD(new_weights, a.__data_shape.copy(), new_node, a.__index_order.copy())


    def contract(self, data_indices: Sequence[Sequence[int]], sum_dict_cache: Dict= None) -> TDD:
        '''
            Contract the tdd according to the specified data_indices. Return the reduced result.
            data_indices should be counted in the data indices only.
            e.g. ([a,b,c],[d,e,f]) means contracting indices a-d, b-e, c-f (of course two lists should be in the same size)
        '''
        if len(data_indices[0]) == 0:
            return self.clone()
        else:
            #transform to inner indices
            reversed_order = order_inverse(self.__index_order)
            inner_ls1 = [reversed_order[data_indices[0][k]] for k in range(len(data_indices[0]))]
            inner_ls2 = [reversed_order[data_indices[1][k]] for k in range(len(data_indices[0]))]

            #inner_ls1[i] < inner_ls2[i] should hold for every i
            node, weights = weighted_node.contract((self.__node, self.__weights),
                     self.__inner_data_shape, [inner_ls1, inner_ls2], sum_dict_cache)

            #assume data_indices[0] and data_indices[1] are in the same type
            new_data_shape, new_index_order = self.__index_reduce_proc(inner_ls1+inner_ls2)

            return TDD(weights, new_data_shape, node, new_index_order)


    def permute(self, dims: Sequence[int]) -> TDD:
        '''
            Returns a view of the original tensor input with its dimensions permuted.
        '''
        res = self.get_view()
        for i in range(len(dims)):
            res.__data_shape[i] = self.__data_shape[dims[i]]
        res.__index_order = [dims[i] for i in self.__index_order]
        return res




    def show(self, path: str='output', real_label: bool=True, full_output: bool = False, precision: int = 2):
        '''
            full_output: if True, then the edge will appear as a tensor, not the parallel index shape.

            (NO TYPING SYSTEM VERIFICATION)
        '''
        edge=[]              
        dot=Digraph(name='reduced_tree')
        dot=Node.layout(self.__node,self.parallel_shape,self.__index_order, dot,edge, real_label, full_output)
        dot.node('-0','',shape='none')

        if self.__node == None:
            id_str = str(TERMINAL_ID)
        else:
            id_str = str(self.__node.id)

        if list(self.__weights.shape)==[2]:
            dot.edge('-0',id_str,color="blue",label=
                str(complex(round(self.__weights[0].cpu().item(),precision),round(self.__weights[1].cpu().item(),precision))))
        else:
            if full_output == True:
                label = str(CUDAcpl2np(self.__weights))
            else:
                label =str(self.parallel_shape)
            dot.edge('-0',id_str,color="blue",label = label)
        dot.format = 'png'
        return Image(dot.render(path))
    


