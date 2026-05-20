import copy
import math
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import random
import os


class IMVDataset(Dataset):
    def __init__(self, imv_data, mask_matrix, labels, num_views):
        self.num_views = num_views
        self.imv_data = imv_data
        self.mask = mask_matrix
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        items = [torch.from_numpy(dv[index]).float() for dv in self.imv_data]
        items.append(torch.from_numpy(self.mask[index]).float())
        items.append(torch.from_numpy(np.array(self.labels[index])).float())
        return items


class SVDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()


def get_mask(num_views, data_size, missing_rate,initialtype,t,dataset_name):
    assert num_views >= 2
    miss_sample_num = math.floor(data_size * missing_rate)
    data_ind = list(range(data_size))
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, num_views])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(num_views)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[miss_ind[j]] = rand_v

    return mask

def save_mask(mask, filepath):
    np.save(filepath, mask)
    print(f"Mask saved to {filepath}")

def load_mask(filepath):
    mask = np.load(filepath)
    print(f"Mask loaded from {filepath}, shape: {mask.shape}")
    return mask


def load_data(args):

    if args.dataset_name == 'Caltech7-5V' or args.dataset_name == 'Multi-Fashion' or args.dataset_name == 'NoisyMNIST' or args.dataset_name == 'Scene-15':
        dataset_dir_base = './data/'
        data_path = dataset_dir_base + args.dataset_name + '.npz'
        data = np.load(data_path)
        num_views = int(data['n_views'])
        data_list = [data[f'view_{v}'].astype(np.float32) for v in range(num_views)]
        labels = data['labels']
        dims = [dv.shape[1] for dv in data_list]
        data_size = labels.shape[0]
        class_num = len(np.unique(labels))
        if np.max(labels) == class_num:
            labels = labels - 1
        args.multiview_dims = dims
        args.num_views = num_views
        args.class_num = class_num
        args.data_size = data_size
        print('data_list',data_list[0].shape)
        print('args.class_num',args.class_num)
        print('args.data_size',args.data_size)
        print('args.num_views',args.num_views)
        print('args.dims',args.multiview_dims)
        print(labels)
        return data_list, labels


    elif args.dataset_name == 'Handwritten':
 
        data_path = './data/Handwritten/'
        
        view_names = [
            'mfeat-fou',  # 76 Fourier coefficients
            'mfeat-fac',  # 216 profile correlations
            'mfeat-kar',  # 64 Karhunen-Love coefficients
            'mfeat-zer',  # 47 Zernike moments
            'mfeat-pix',  # 240 pixel averages
            'mfeat-mor'   # 6 morphological features
        ]
        
 
        origin_mv_data = []
        for view_name in view_names:
            data = np.loadtxt(os.path.join(data_path, view_name), dtype=np.float32)
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            origin_mv_data.append(data)

        labels = np.repeat(np.arange(10), 200).astype(np.float32)
 
        data_size = 2000   
        dims = [76, 216, 64, 47, 240, 6]  
        class_num = 10 
        num_views = 6  
        
 
        args.multiview_dims = dims
        args.num_views = num_views
        args.class_num = class_num
        args.data_size = data_size
        print('args.class_num',args.class_num)
        print('args.data_size',args.data_size)
        print('args.num_views',args.num_views)
        print('args.dims',args.multiview_dims)
        print(labels)

        return origin_mv_data, labels

    elif args.dataset_name == 'CUB_600':
        # Load CUB dataset from .mat file
        data_path = './data/cub_googlenet_doc2vec_c10.mat'
        mat_data = scipy.io.loadmat(data_path)
        

        view0 = mat_data['X'][0, 0].T.astype(np.float32) 
        view1 = mat_data['X'][0, 1].T.astype(np.float32)  
        
        origin_mv_data = [view0, view1]
        
 
        labels = mat_data['gt'].flatten().astype(np.int64) 
        labels = labels - 1
        
        data_size = labels.shape[0]
        dims = [1024, 300]
        class_num = 10
        num_views = 2
        
        args.multiview_dims = dims
        args.num_views = num_views
        args.class_num = class_num
        args.data_size = data_size

        return origin_mv_data, labels
def pixel_normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)

def build_dataset(args,t):
    origin_mv_data, labels = load_data(args)

    if args.dataset_name == 'Caltech7-5V' or args.dataset_name == 'Multi-Fashion':
        origin_mv_data = [pixel_normalize(dv) for dv in origin_mv_data]

    if args.dataset_name == 'Scene-15':
        origin_mv_data = [StandardScaler().fit_transform(sv_data) for sv_data in origin_mv_data]
    
    
    mask = get_mask(args.num_views, args.data_size, args.missing_rate,args.initial_str,t,args.dataset_name).astype(np.float32)

    imv_data = [origin_mv_data[v] * mask[:, v:v + 1] for v in range(args.num_views)]

    imv_dataset = IMVDataset(imv_data, mask, labels, args.num_views)
    imv_loader = DataLoader(imv_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,
                        drop_last=False)

    com_idx = np.sum(mask, axis=1) == args.num_views
    cmv_data = [sv_data[com_idx] for sv_data in imv_data]

    sv_datasets = [SVDataset(copy.deepcopy(imv_data[v][mask[:, v] == 1])) for v in range(args.num_views)]
    sv_loaders = [DataLoader(sv_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            drop_last=False) for sv_dataset in sv_datasets]

    return cmv_data, imv_loader, sv_loaders
