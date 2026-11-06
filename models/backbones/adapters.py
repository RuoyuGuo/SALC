import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
# from einops import rearrange
import einops


class DeltaBlock(nn.Module):
    '''
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    
    '''
    
    def __init__(self, dim, bottleneck_dim, dropout_rate=0.1, borrow_flag=True):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)
        self.borrow_flag = borrow_flag
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)
            
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        
    def forward(self, x_self, x_other):
        # if self.borrow_flag:
        delta = self.down(x_other)
        # else:
        #     delta = self.down(x_self)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        return delta