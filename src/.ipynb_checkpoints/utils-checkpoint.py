import os, sys, torch, random, numpy as np
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import RandomSampler
from matplotlib import pyplot as plt
from src.dataset import CustomDataset
sys.path.append("./src")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def get_dls(train_df, valid_df, test_df, meta_features, train_tfs, valid_tfs, bs, nws):
    
    # Datasets
    train_ds = CustomDataset(df = train_df, mode = 'train', meta_features = meta_features, transformations = train_tfs)
    valid_ds = CustomDataset(df = valid_df, mode = 'valid', meta_features = meta_features, transformations = valid_tfs)
    test_ds  = CustomDataset(df = test_df,  mode = 'test', meta_features = meta_features, transformations = valid_tfs)
    
    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds), num_workers = nws)
    valid_dl = DataLoader(valid_ds, batch_size = bs, num_workers = nws)
    test_dl  = DataLoader(test_ds, batch_size = bs, num_workers = nws)
    
    return train_dl, valid_dl, test_dl
    
def makedirs(dir_list): [os.makedirs(dir_name, exist_ok = True) for dir_name in dir_list]

def plot_learning_curves(plot_data, title, save_name, save_path, loss_data = False):
    if loss_data: plt.plot(np.array(plot_data)[:, 0], label = "Train Loss"); plt.plot(np.array(plot_data)[:, 1], label = "Validation Loss")
    else: plt.plot(plot_data, label = title)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(title); plt.legend()
    plt.xticks(ticks = np.arange(len(plot_data)), labels = np.arange(1, len(plot_data) + 1))
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()
        