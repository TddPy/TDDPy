from __future__ import annotations
from typing import Iterable, Tuple, List, Any, Union, cast
import numpy as np
import torch

from . import CUDAcpl
from .CUDAcpl import _U_, CUDAcpl_Tensor, CUDAcpl2np
from . import node
from .node import  TERMINAL_ID, Node, IndexOrder, order_inverse
import copy


from graphviz import Digraph
from IPython.display import Image




class TDD:
    '''
        TDD  functions as the compact representation of tensors,
        and can fit into tensor networks.
    '''
    def __init__(self, 
                    weights: CUDAcpl_Tensor,
                    data_shape: List[int],
                    node: Node|None,
                    index_order: IndexOrder = []):
        self.weights: CUDAcpl_Tensor = weights
        self.data_shape: List[int] = data_shape  #the data index shape
        self.node: Node|None = node

        '''
            index_order: how the inner index are mapped to outer representations
            for example, tdd[a,b,c] under index_order=[0,2,1] returns the value tdd[a,c,b]
            index_order == None means the trival index mapping [0,1,2,(...)]
        '''
        self.index_order: IndexOrder = index_order
    @property
    def parallel_shape(self) -> List[int]:
        return list(self.weights.shape[:-1])

    @property
    def global_order(self)-> List[int]:
        '''
            Return the index order containing both parallel and data indices.
            Note that the last index reserved for CUDA complex is not included
        '''
        parallel_index_order = [i for i in range(len(self.parallel_shape))]
        increment = len(self.parallel_shape)
        return parallel_index_order + [order+increment for order in self.index_order]

    def __eq__(self, other: TDD) -> bool:
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
    def __construct_and_normalize(order, the_successors: List[TDD]):
        '''
            construct the tdd with the_successors, and normalize it

            order: represent the order of this node (which tensor index it represent)

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
            
            node=None

            res=TDD(weights,[],node,[])
            return res

        for k in range(len(the_successors)):
            if the_successors[k].data_shape != the_successors[0].data_shape:
                raise Exception('This method requires the data shape of the_successors to be the same.')
            if the_successors[k].index_order != the_successors[0].index_order:
                raise Exception('This method requires the index order of the_successors to be the same.')

            int_key = Node.get_int_key(weigs[k])
            if torch.max(int_key) == 0 and torch.min(int_key) == 0:
                weights = _U_(torch.zeros_like,weigs[k])
                node=None
                the_successors[k]=TDD(weights,[],node,[])

                weigs[k] = _U_(torch.zeros_like,weigs[k])
        
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
        succ_nodes=[succ.node for succ in the_successors]

        node=Node.get_unique_node(order,weigs,succ_nodes)
        res=TDD(weig_max,[],node,[])
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
            node = None
            res = TDD(weights,[],node,[])
            return res
        

        split_pos=index_order[depth]
        split_tensor = list(tensor.split(1,-len(data_shape)+split_pos-1))
            #-1 is because the extra inner dim for real and imag

        the_successors: List[TDD] =[]

        for k in range(data_shape[split_pos]):
            res = TDD.__as_tensor_iterate(split_tensor[k],parallel_shape,index_order,depth+1)
            the_successors.append(res)

        tdd = TDD.__construct_and_normalize(depth,the_successors)
        return tdd


    @staticmethod
    def as_tensor(data : Union[CUDAcpl_Tensor,np.ndarray,Tuple]) -> TDD:
        '''
        construct the tdd tensor

        tensor:
            1. in the form of a matrix only: assume the parallel index and index order to be []
            2. in the form of a tuple (data, index_shape, index_order)
            Note that if the input matrix is a torch tensor, 
                    then it must be already in CUDAcpl_Tensor(CUDA complex) form.
        '''

        if isinstance(data,Tuple):
            tensor,parallel_shape,index_order = data
        else:
            tensor = data
            parallel_shape = []
            index_order: List[int] = []
            
        if isinstance(tensor,np.ndarray):
            tensor = CUDAcpl.np2CUDAcpl(tensor)

        #pre-process above

        data_shape = list(tensor.shape[len(parallel_shape):-1])  #the data index shape
        if index_order == []:
            result_index_order = list(range(len(data_shape)))
        else:
            result_index_order = index_order.copy()


        if len(data_shape)!=len(result_index_order):
            raise Exception('The number of indices must match that provided by tensor.')

        '''
            This extra layer is also for copying the input list and pre-process.
        '''
        res = TDD.__as_tensor_iterate(tensor,parallel_shape,result_index_order,0)

        
        res.index_order = result_index_order
        res.data_shape = data_shape
        return res

    
            
    def CUDAcpl(self) -> CUDAcpl_Tensor:
        '''
            Transform this tensor to a CUDA complex and return.
        '''
        trival_ordered_data_shape = [self.data_shape[i] for i in order_inverse(self.index_order)]
        node_data = Node.CUDAcpl_Tensor(self.node,self.weights,trival_ordered_data_shape)
        
        #permute to the right index order
        node_data = node_data.permute(tuple(self.global_order+[node_data.dim()-1]))

        expanded_weights = self.weights.view(tuple(self.parallel_shape)+(1,)*len(self.data_shape)+(2,))
        expanded_weights = expanded_weights.expand_as(node_data)

        return CUDAcpl.einsum('...,...->...',node_data,expanded_weights)
        

    def numpy(self) -> np.ndarray:
        '''
            Transform this tensor to a numpy ndarry and return.
        '''
        return CUDAcpl2np(self.CUDAcpl())



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
    




    def show(self,real_label: bool=True,path: str='output', full_output: bool = False):
        '''
            full_output: if True, then the edge will appear as a tensor, not the parallel index shape.

            (NO TYPING SYSTEM VERIFICATION)
        '''
        edge=[]              
        dot=Digraph(name='reduced_tree')
        dot=Node.layout(self.node,self.parallel_shape,self.index_order, dot,edge, real_label, full_output)
        dot.node('-0','',shape='none')

        if self.node == None:
            id_str = str(TERMINAL_ID)
        else:
            id_str = str(self.node.id)

        if list(self.weights.shape)==[2]:
            dot.edge('-0',id_str,color="blue",label=
                str(complex(round(self.weights[0].cpu().item(),2),round(self.weights[1].cpu().item(),2))))
        else:
            if full_output == True:
                label = str(CUDAcpl2np(self.weights))
            else:
                label =str(self.parallel_shape)
            dot.edge('-0',id_str,color="blue",label = label)
        dot.format = 'png'
        return Image(dot.render(path))