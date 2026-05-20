import torch
from evaluate import evaluate
import numpy as np
from config import device  


def train(model, dvimc_loss, optimizer, scheduler, imv_loader):
    model.train()

    for batch_idx, item in enumerate(imv_loader):
        data_list = [data.to(device) for data in item[:-2]]
        mask_matrix = item[-2].to(device)
        
        optimizer.zero_grad()
        _, vs_mus, vs_stds, aggregated_mu, aggregated_var, xr_list, vade_z_sample = model(data_list, mask_matrix)
        vs_vars = [std ** 2 for std in vs_stds]
 
        total_loss = dvimc_loss(vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample, 
                               model.prior_weight, model.prior_mu, model.prior_std, data_list, mask_matrix, model.corr)        
        total_loss.backward()
 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)

        optimizer.step()
        
        with torch.no_grad():
            model.prior_weight.data = model.prior_weight.data / torch.sum(model.prior_weight.data, dim=0, keepdim=True)
            
    scheduler.step()

    return 


def log_gaussian(x, mu, var):
    return -0.5 * (torch.log(torch.tensor(2.0 * np.pi, device=x.device)) + torch.log(var) + torch.pow(x - mu, 2) / var)


def mog_predict(mu, mog_weight, mog_mu, mog_var):
    log_pz_c = torch.sum(log_gaussian(mu.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1)
    log_pc = torch.log(mog_weight.unsqueeze(0))
    log_pc_z = log_pc + log_pz_c
    pc_z = torch.exp(log_pc_z) + 1e-10
    normalized_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)
    return normalized_pc_z


def test(model, imv_loader):
    model.eval()
    c_assignment = []
    true_labels = []

    with torch.no_grad():
        for _, item in enumerate(imv_loader):
            imv_data = [data.to(device) for data in item[:-2]]
            mask = item[-2].to(device)
            labels = item[-1]
            _, _, _, aggregated_mu, aggregated_var, _, _ = model(imv_data, mask)
            mog_weight, mog_mu, mog_std = model.prior_weight, model.prior_mu, model.prior_std
            mog_var = mog_std**2
            c_assignment.append(mog_predict(aggregated_mu, mog_weight, mog_mu, mog_var))
            true_labels.append(labels)

    true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
    c_assignment = torch.cat(c_assignment, dim=0)
    predict = torch.argmax(c_assignment, dim=1).cpu().numpy()
    acc, nmi, ari, pur = evaluate(true_labels, predict)

    return acc, nmi, ari, pur
