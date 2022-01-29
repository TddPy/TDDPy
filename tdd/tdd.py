import numpy as np
import torch

from . import CUDAcpl
from .CUDAcpl import _U_
import copy


from graphviz import Digraph
from IPython.display import Image

from .global_val import EPS

class TDD:
    pass