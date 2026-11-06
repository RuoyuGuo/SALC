import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss

@LOSSES.register_module()
class RateLoss(nn.Module):
    def __init__(self, 
                 weight=None, 
                 reduction='mean',
                 loss_name='loss_rate',
                 loss_weight=1.0,
                 
                 # loss args: larger to keep more tokens
                 # lower to keep less tokens
                 token_target_ratio_vlm=0.5,
                 token_target_ratio_vfm=0.5,
                 ) -> None:
        super().__init__()
        self.token_target_ratio_vlm = token_target_ratio_vlm
        self.token_target_ratio_vfm = token_target_ratio_vfm
        
        self.weight = weight
        self.reduction = reduction
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        
    def forward(self, pred, target, model, **kwargs):
        if model == 'vlm':
            token_target_ratio = self.token_target_ratio_vlm
        if model == 'vfm':
            token_target_ratio = self.token_target_ratio_vfm
            
        pred = pred.float()
        token_select = pred
        # (bs, num_layers, L, 1)
        
        
        # print('This is loss', token_select.shape)
        
        token_mean = token_select.mean(dim=(1, 2, 3))
        token_flops_loss = (token_mean - token_target_ratio)**2
        
        loss = weight_reduce_loss(token_flops_loss, self.weight, reduction=self.reduction)
        
        return loss * self.loss_weight

@LOSSES.register_module()
class CKALoss(nn.Module):
    def __init__(self,                  
                 weight=None, 
                 reduction='mean',
                 loss_name='loss_red',
                 loss_weight=1.0,

                 #
                 layers=None,
                ) -> None:

        super().__init__()
        self.layers = layers
        self.weight = weight
        self.reduction = reduction
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        
        
    
    # def cka_linear(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    def cka(self, X, Y, eps=1e-8):
        assert X.shape == Y.shape and X.dim() == 4
        B, L, T, D = X.shape
    
        # Flatten over (B,L) so we can use bmm
        Xf = X.reshape(B*L, T, D)
        Yf = Y.reshape(B*L, T, D)
    
        # Center across tokens per (B,L)
        Xc = Xf - Xf.mean(dim=1, keepdim=True)     # [BL,T,D]
        Yc = Yf - Yf.mean(dim=1, keepdim=True)     # [BL,T,D]
    
        # Covariances via batched matmul: [BL,D,D]
        Xt = Xc.transpose(1, 2).contiguous()
        Yt = Yc.transpose(1, 2).contiguous()
        XtY = torch.bmm(Xt, Yc)   # X^T Y
        XtX = torch.bmm(Xt, Xc)   # X^T X
        YtY = torch.bmm(Yt, Yc)   # Y^T Y
    
        # ||X^T Y||_F^2 and norms per (B,L)
        num = (XtY * XtY).sum(dim=(1, 2))                                # [BL]
        den = torch.linalg.norm(XtX, dim=(1, 2)) * torch.linalg.norm(YtY, dim=(1, 2)) + eps  # [BL]
        cka_bl = (num / den).reshape(B, L)                                # [B,L]
        
        if self.layers is not None:
            cka_bl = cka_bl[:, self.layers]
        
        loss = 1.0 - cka_bl.mean()
        return cka_bl, loss
    
    def forward(self, X, Y, **kwargs):
        '''
        shape, B * layer * token * dim
        '''
        loss_dict = {}
        #compute token level
        _, cka_loss = self.cka(X, Y)

        loss_dict['token_cka'] = cka_loss * self.loss_weight
        
        return loss_dict

@LOSSES.register_module()
class CCKALoss(nn.Module):
    def __init__(self,                  
                 weight=None, 
                 reduction='mean',
                 loss_name='loss_red',
                 loss_weight=1.0,

                 #
                 layers=None,
                ) -> None:

        super().__init__()
        self.layers = layers
        self.weight = weight
        self.reduction = reduction
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        
        
    
    # def cka_linear(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    def cka(self, X, Y, eps=1e-8):
        assert X.shape == Y.shape and X.dim() == 4
        B, L, T, D = X.shape
    
        # Flatten over (B,L) so we can use bmm
        Xf = X.reshape(B*L, T, D)
        Yf = Y.reshape(B*L, T, D)
    
        # Center across tokens per (B,L)
        Xc = Xf - Xf.mean(dim=1, keepdim=True)     # [BL,T,D]
        Yc = Yf - Yf.mean(dim=1, keepdim=True)     # [BL,T,D]
    
        # Covariances via batched matmul: [BL,D,D]
        Xt = Xc.transpose(1, 2).contiguous()
        Yt = Yc.transpose(1, 2).contiguous()
        XtY = torch.bmm(Xt, Yc)   # X^T Y
        XtX = torch.bmm(Xt, Xc)   # X^T X
        YtY = torch.bmm(Yt, Yc)   # Y^T Y
    
        # ||X^T Y||_F^2 and norms per (B,L)
        num = (XtY * XtY).sum(dim=(1, 2))                                # [BL]
        den = torch.linalg.norm(XtX, dim=(1, 2)) * torch.linalg.norm(YtY, dim=(1, 2)) + eps  # [BL]
        cka_bl = (num / den).reshape(B, L)                                # [B,L]
        
        if self.layers is not None:
            cka_bl = cka_bl[:, self.layers]
        
        loss = cka_bl.mean()
        return cka_bl, loss
    
    def forward(self, X, Y, **kwargs):
        '''
        shape, B * layer * token * dim
        '''
        loss_dict = {}
        #compute token level
        _, cka_loss = self.cka(X, Y)

        loss_dict['token_cka'] = cka_loss * self.loss_weight
        
        return loss_dict
        
        

@LOSSES.register_module()
class SelfCKALoss(nn.Module):
    def __init__(self,                  
                 weight=None, 
                 reduction='mean',
                 loss_name='loss_red',
                 loss_weight=1.0,

                 #
                 vlmlayers=None,
                 vfmlayers=None,
                ) -> None:

        super().__init__()
        self.vlmlayers = vlmlayers
        self.vfmlayers = vfmlayers
        self.weight = weight
        self.reduction = reduction
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        
        
    
    # def cka_linear(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    def cka(self, X, Y, model=None):
        assert X.shape == Y.shape and X.dim() == 4
        eps=1e-8
        B, L, T, D = X.shape
    
        # Flatten over (B,L) so we can use bmm
        Xf = X.reshape(B*L, T, D)
        Yf = Y.reshape(B*L, T, D)
    
        # Center across tokens per (B,L)
        Xc = Xf - Xf.mean(dim=1, keepdim=True)     # [BL,T,D]
        Yc = Yf - Yf.mean(dim=1, keepdim=True)     # [BL,T,D]
    
        # Covariances via batched matmul: [BL,D,D]
        Xt = Xc.transpose(1, 2).contiguous()
        Yt = Yc.transpose(1, 2).contiguous()
        XtY = torch.bmm(Xt, Yc)   # X^T Y
        XtX = torch.bmm(Xt, Xc)   # X^T X
        YtY = torch.bmm(Yt, Yc)   # Y^T Y
    
        # ||X^T Y||_F^2 and norms per (B,L)
        num = (XtY * XtY).sum(dim=(1, 2))                                # [BL]
        den = torch.linalg.norm(XtX, dim=(1, 2)) * torch.linalg.norm(YtY, dim=(1, 2)) + eps  # [BL]
        cka_bl = (num / den).reshape(B, L)                                # [B,L]
        
        if model is not None:
            if model == 'vlm' and self.vlmlayers is not None:
                cka_bl = cka_bl[:, self.vlmlayers]
            elif model == 'vfm' and self.vfmlayers is not None:
                cka_bl = cka_bl[:, self.vfmlayers]
        
        loss = 1-cka_bl.mean()
        return cka_bl, loss
    
    def forward(self, X, Y, model=None, **kwargs):
        '''
        shape, B * layer * token * dim
        '''
        loss_dict = {}
        #compute token level
        _, cka_loss = self.cka(X, Y, model)
        loss_dict['token_cka'] = cka_loss * self.loss_weight

        return loss_dict