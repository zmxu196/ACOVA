import torch
import torch.nn as nn
import numpy as np


class ACOVA_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.likelihood = args.likelihood
        self.alpha = args.alpha
        if self.likelihood == 'Bernoulli':
            self.rec_fn = nn.BCEWithLogitsLoss(reduction='none')

        else:
            self.rec_fn = nn.MSELoss(reduction='none')

    def log_gaussian(self, x, mu, var):
        return -0.5 * (torch.log(torch.tensor(2.0 * np.pi, device=x.device)) + torch.log(var) + torch.pow(x - mu, 2) / var)

    def gaussian_kl(self, q_mu, q_var, p_mu, p_var):
        return -0.5 * (torch.log(q_var / p_var) - q_var / p_var - torch.pow(q_mu - p_mu, 2) / p_var + 1)

    def vade_trick(self, mc_sample, prior_weight, prior_mu, prior_var):
 
        log_pz_c = torch.sum(self.log_gaussian(mc_sample.unsqueeze(1), prior_mu.unsqueeze(0), prior_var.unsqueeze(0)), dim=-1) 
        log_pc = torch.log(prior_weight.unsqueeze(0)) 
        log_pc_z = log_pc + log_pz_c  
        pc_z = torch.exp(log_pc_z) + 1e-10  
        normalized_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)  
        return normalized_pc_z

    def kl_term(self, z_mu, z_var, qc_x, prior_weight, prior_mu, prior_var):
        z_kl_div = torch.sum(qc_x * torch.sum(self.gaussian_kl(z_mu.unsqueeze(1), z_var.unsqueeze(1), 
                                                              prior_mu.unsqueeze(0), prior_var.unsqueeze(0)), dim=-1), dim=1)
        z_kl_div_mean = torch.mean(z_kl_div)

        c_kl_div = torch.sum(qc_x * torch.log(qc_x / prior_weight.unsqueeze(0)), dim=1)
        c_kl_div_mean = torch.mean(c_kl_div)
        return z_kl_div_mean + c_kl_div_mean


    def coherence_term(self, vs_mus, vs_vars, aggregated_mu, aggregated_var, mask=None):
        mv_coherence_loss = []
        norm = torch.sum(mask, dim=1)
        for v in range(len(vs_mus)):
            sv_coherence_loss = torch.sum(self.gaussian_kl(aggregated_mu, aggregated_var, vs_mus[v], vs_vars[v]), dim=1)
            exist_loss = sv_coherence_loss * mask[:, v]
            mv_coherence_loss.append(exist_loss)
        coherence_loss = torch.mean(sum(mv_coherence_loss) / norm) 
        return coherence_loss



    def forward(self, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample, prior_weight, prior_mu, prior_std, imv_data, mask, corr):
        prior_var = prior_std**2
        qc_x = self.vade_trick(vade_z_sample, prior_weight, prior_mu, prior_var)
        kl_loss = self.kl_term(aggregated_mu, aggregated_var, qc_x, prior_weight, prior_mu, prior_var)
        coherence_loss = self.coherence_term(vs_mus, vs_vars, aggregated_mu, aggregated_var, mask)

        mv_rec_loss = []
        for v in range(len(imv_data)):
            rec_loss = torch.sum(self.rec_fn(xr_list[v], imv_data[v]), dim=1)
            exist_rec = rec_loss * mask[:, v]
            sv_rec_loss = torch.mean(exist_rec) 
            mv_rec_loss.append(sv_rec_loss)
 
        rec_loss = torch.mean(sum(mv_rec_loss))

        return (rec_loss + kl_loss) + self.alpha * coherence_loss 
