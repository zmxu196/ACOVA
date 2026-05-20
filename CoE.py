
import torch
import torch.nn as nn
from typing import Tuple
from config import device

class CoE(nn.Module):
    def __init__(self, corr, modalities=5):
        super().__init__()
        self.corr = corr
        self.modalities = modalities
        self.corr_matrix = self.corr
        
    @torch.jit.export
    def get_coe_distribution_params(self, mask, mu_experts, std_experts, eps=1e-5):
 
        R = self.corr_matrix   
        
        imv_mask = mask.unsqueeze(1)          
        mu_masked = mu_experts * imv_mask     
        std = std_experts.unsqueeze(-1)      

        D = std                                                
        Sigma = D @ D.transpose(-1, -2) * R[None, None, :, :]  

         
        inv_cov = torch.inverse(Sigma)                 
        
        m = mask.unsqueeze(1).unsqueeze(-1)                  
        mask_mat = torch.matmul(m, m.transpose(-1, -2))      
        inv_cov = inv_cov * mask_mat                         

        sum_inv = inv_cov.sum(dim=(2, 3), keepdim=True)                  
        weights = inv_cov.sum(dim=3) / (sum_inv.squeeze(-1) + eps)       
 
        mu_coe = torch.sum(weights * mu_masked, dim=2)           
        var_coe = 1. / (sum_inv.squeeze(-1).squeeze(-1) + eps)   
        
        return mu_coe, var_coe

