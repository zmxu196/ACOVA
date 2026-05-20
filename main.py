import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.cluster import KMeans
import numpy as np
import random
import argparse
from config import device   
from datasets import build_dataset
from acova_model import ACOVA_model
from acova_loss import ACOVA_loss
from run_epoch import train, test




def safe_cholesky(matrix, eps=1e-5):
    M = (matrix + matrix.T) / 2
    eigvals = torch.linalg.eigvalsh(M)
    min_eig = torch.min(eigvals)
    if min_eig < eps:
        M = M + torch.eye(M.size(0), device=M.device) * (eps - min_eig + 1e-6)
    return torch.linalg.cholesky(M)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def initialization(model, cmv_data, sv_loaders, args):
    print('Initializing......')
    model.train()
    criterion = nn.MSELoss()

    for v in range(args.num_views):
        networks_parameters = [p for name, p in model.named_parameters() 
                             if f'view_{v}' in name and 'std' not in name]
        optimizer = Adam(networks_parameters)
        
        for e in range(1, args.initialization_epochs + 1):
            for xv in sv_loaders[v]:
                xv = xv.to(device)
                optimizer.zero_grad()
                _, sv_rec = model.vs_encode(xv, v)
                vs_rec_loss = criterion(sv_rec, xv)
                vs_rec_loss.backward()
                optimizer.step()
    model.eval()
    with torch.no_grad():
        fit_data = [torch.tensor(csv_data, dtype=torch.float32).to(device) for csv_data in cmv_data]
        latent_representations = []
        for v in range(args.num_views):
            latent, _ = model.vs_encode(fit_data[v], v)
            latent_representations.append(latent)

        fused_latent_representations = sum(latent_representations) / len(latent_representations) 

        kmeans = KMeans(n_clusters=args.class_num, n_init=10)
        kmeans.fit(fused_latent_representations.cpu().numpy())
        model.prior_mu.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

        if args.initMatrix == True:
            if args.calSimtype == 'cosine':
                similarity_matrix = torch.zeros((args.num_views, args.num_views), device=device)
                for i in range(args.num_views):
                    for j in range(args.num_views):
                        if i == j:
                            similarity_matrix[i, j] = 1.0
                        else:
                            similarity_matrix[i, j] = torch.cosine_similarity(latent_representations[i], latent_representations[j], dim=1).mean()

            elif args.calSimtype == 'Pearson':
                latent_tensor = torch.stack(latent_representations)  
                latent_flat = latent_tensor.reshape(args.num_views, -1)   
                similarity_matrix = torch.corrcoef(latent_flat)   

            elif args.calSimtype == 'identity':
                    model.corr_param.data = torch.eye(args.num_views, device=device)
                    return 
            elif args.calSimtype == 'randn':
                model.corr_param.data = torch.randn(args.num_views, args.num_views, device=device)
                return
            elif args.calSimtype == 'absrandn':
                model.corr_param.data = torch.abs(torch.randn(args.num_views, args.num_views, device=device))
                return
            elif args.initial_str == 'safe_cholesky':
                print('safe_cholesky...')
                L0 = safe_cholesky(similarity_matrix)
                model.corr_param.data = L0
                return


def main(args):


    for i in range(1, args.test_times + 1):
        t = i
        # mask generation seed
        np.random.seed(t)
        random.seed(t)
        cmv_data, imv_loader, sv_loaders = build_dataset(args,t)
        # training seed
        setup_seed(args.seed) 
        model = ACOVA_model(args).to(device)
        
        networks_parameters = [p for name, p in model.named_parameters() 
                            if 'encoders' in name or 'decoders' in name]
        prior_parameters = [p for name, p in model.named_parameters() 
                        if 'prior' in name]
        corr_parameters = [p for name, p in model.named_parameters() if 'corr' in name]

        optimizer = Adam([
            {'params': networks_parameters, 'lr': args.learning_rate},
            {'params': prior_parameters, 'lr': args.prior_learning_rate},
            {'params': corr_parameters, 'lr': args.corr_learning_rate} 
        ], betas=(0.9, 0.999), eps=1e-8)
        
        decay_rate = args.lr_decay_gamma ** (1 / args.lr_decay_step)
        scheduler = ExponentialLR(optimizer, gamma=decay_rate)
        
        initialization(model, cmv_data, sv_loaders, args)
        acova_loss = ACOVA_loss(args)

        print('training...')

        for epoch in range(args.train_epochs):

            train(model, acova_loss, optimizer, scheduler, imv_loader)

            if epoch % args.log_interval == 0:  
                acc, nmi, ari, pur = test(model, imv_loader)
                print(f'Epoch {epoch:>3}/{args.train_epochs} '
                    f'ACC : {acc:.4f} NMI: {nmi:.4f} ARI: {ari:.4f} PUR: {pur:.4f}')
                
        final_results = test(model, imv_loader)
        print(f'Final Results after {args.train_epochs} epochs : ACC {final_results[0]:.4f} NMI {final_results[1]:.4f} '
            f'ARI {final_results[2]:.4f} PUR {final_results[3]:.4f}')
        print("#"*30)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', type=int, default=300, help='training epochs') 
    parser.add_argument('--initialization_epochs', type=int, default=200, help='initialization epochs') 
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')  
    parser.add_argument('--prior_learning_rate', type=float, default=0.05, help='prior learning rate')
    parser.add_argument('--corr_learning_rate', type=float, default=0.01, help='corr learning rate')
    
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=float, default=20, help='StepLr_Step_size') 
    parser.add_argument('--lr_decay_gamma', type=float, default=0.9, help='StepLr_Gamma') 

    parser.add_argument('--dataset', type=int, default=5, help='0:Caltech7-5v, 1:Scene-15, 2:Multi-Fashion, 3:NoisyMNIST, 4:Handwritten,5:CUB')
    parser.add_argument('--log_interval', type=int, default=50)

    parser.add_argument('--test_times', type=int, default=1, help='test times')
 
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=10) 
    parser.add_argument('--initMatrix', type=bool, default=True) 
    parser.add_argument('--calSimtype', default='cosine', help='cosine, Pearson')
    parser.add_argument('--initial_str', default='safe_cholesky', help='safe_cholesky, identity, randn, absrandn')
    args = parser.parse_args()
    args.dataset_dir_base = "./data/"
    


    if args.dataset == 0:
        args.dataset_name = 'Caltech7-5V'
        args.alpha = 5
        args.likelihood = 'Gaussian'
        args.learning_rate = 0.001
        args.prior_learning_rate = 0.1 #0.05
        args.corr_learning_rate = 0.01
        args.lr_decay_step = 20
        args.lr_decay_gamma = 0.95
        args.z_dim = 15
        args.seed = 1  #2
        args.initial_str = 'randn'


    if args.dataset == 1:
        args.dataset_name = 'Scene-15'
        args.likelihood = 'Gaussian'
        args.train_epochs = 300
        args.initialization_epochs = 200
        args.seed = 1
        args.prior_learning_rate = 0.05
        args.corr_learning_rate = 0.01
        args.z_dim = 10
        args.learning_rate = 0.001   
        args.alpha = 20
        args.lr_decay_step = 20
        args.lr_decay_gamma = 0.9

    if args.dataset == 2:
        args.dataset_name = 'Multi-Fashion'
        args.likelihood = 'Gaussian'
        args.seed = 2 
        args.prior_learning_rate = 0.05
        args.corr_learning_rate = 0.01
        args.z_dim = 10
        args.learning_rate = 0.001   
        args.alpha = 20


    if args.dataset == 3:
        args.dataset_name = 'NoisyMNIST'
        args.alpha = 10 
        args.batch_size = 512
        args.likelihood = 'Bernoulli'
        args.seed = 10
        args.learning_rate = 0.001
        args.prior_learning_rate = 0.05
        args.corr_learning_rate = 0.01

    if args.dataset == 4:
        args.dataset_name = 'Handwritten'
        args.alpha = 15
        args.likelihood = 'Gaussian'
        args.learning_rate = 0.0003
        args.prior_learning_rate = 0.01
        args.corr_learning_rate = 0.01
        args.z_dim = 10
        args.seed = 19
        args.lr_decay_step = 50 #20
        args.lr_decay_gamma = 0.9 #0.95

    if args.dataset == 5:
        args.dataset_name = 'CUB_600'
        args.alpha = 60
        args.seed = 42
        args.likelihood = 'Gaussian'
        args.batch_size = 256
        args.learning_rate = 0.0005  
        args.prior_learning_rate = 0.05
        args.corr_learning_rate = 0.01  
        args.z_dim = 10
        args.train_epochs = 300
        args.initialization_epochs = 200
        args.lr_decay_step = 10
        args.lr_decay_gamma = 0.9

    main(args)
