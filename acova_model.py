import torch
import torch.nn as nn
import torch.distributions as dist
from CoE import CoE
from config import device  




class view_specific_encoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super().__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU()
        )
        self.z_mu = nn.Linear(2000, self.z_dim)
        self.z_std = nn.Linear(2000, self.z_dim)


    def forward(self, x):
        hidden_feature = self.encoder(x)
        zv_mu = self.z_mu(hidden_feature)
        zv_std = torch.nn.functional.softplus(self.z_std(hidden_feature))
        return zv_mu, zv_std


class view_specific_decoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super().__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, self.x_dim)
        )

    def forward(self, z):
        xr = self.decoder(z)
        return xr



class ACOVA_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims
        self.k = args.class_num
        self.z_dim = args.z_dim
        self.num_views = args.num_views

        self.register_parameter('prior_weight', nn.Parameter(torch.full((self.k,), 1/self.k)))
        self.register_parameter('prior_mu', nn.Parameter(torch.zeros(self.k, self.z_dim)))
        self.register_parameter('prior_std', nn.Parameter(torch.ones(self.k, self.z_dim)))

        triled_matrix_later = torch.abs(torch.randn(self.num_views, self.num_views))

        self.register_parameter('corr_param', nn.Parameter(triled_matrix_later))
        self.encoders = nn.ModuleDict({
            f'view_{v}': view_specific_encoder(self.x_dim_list[v], self.z_dim) 
            for v in range(args.num_views)
        })

        self.decoders = nn.ModuleDict({
            f'view_{v}': view_specific_decoder(self.x_dim_list[v], self.z_dim) 
            for v in range(args.num_views)
        })


    @property
    def corr(self):
        L = torch.tril(self.corr_param)
        diag_indices = torch.arange(self.num_views, device=self.corr_param.device)
        L[diag_indices, diag_indices] = torch.exp(L[diag_indices, diag_indices])
        cov = L @ L.T
        d = torch.sqrt(torch.diag(cov))
        R = cov / (d[:, None] * d[None, :])
  
        return R
        
    def inference_z(self, imv_data, mask=None):
        vs_mus, vs_stds = [], []
        for v in range(self.num_views):
            vs_mu, vs_std = self.encoders[f'view_{v}'](imv_data[v]) 
            vs_mus.append(vs_mu)
            vs_stds.append(vs_std)
        mu = torch.stack(vs_mus)  
        std = torch.stack(vs_stds) 
        mus = mu.permute(1, 2, 0)   
        stds = std.permute(1, 2, 0)  
        
        if len(mask.shape) == 2: 
            mask = mask.float().to(device) 

        coe = CoE(corr=self.corr, modalities=self.num_views)
        coe_mu, coe_var = coe.get_coe_distribution_params(mask, mus, stds)
        aggregated_mu = coe_mu
        aggregated_var = coe_var
    
        return vs_mus, vs_stds, aggregated_mu, aggregated_var

    def generation_x(self, z):
        xr_list = [self.decoders[f'view_{v}'](z) for v in range(self.num_views)]
        return xr_list

    def sample(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, imv_data, mask=None):
        vs_mus, vs_stds, aggregated_mu, aggregated_var = self.inference_z(imv_data, mask)
        z_sample = self.sample(aggregated_mu, aggregated_var)
        xr_list = self.generation_x(z_sample)
        vade_z_sample = self.sample(aggregated_mu, aggregated_var)
        return z_sample, vs_mus, vs_stds, aggregated_mu, aggregated_var, xr_list, vade_z_sample

    def vs_encode(self, sv_data, view_idx):
        latent_representation, _ = self.encoders[f'view_{view_idx}'](sv_data)
        xv_rec = self.decoders[f'view_{view_idx}'](latent_representation)
        return latent_representation, xv_rec


