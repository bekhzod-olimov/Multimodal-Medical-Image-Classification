import os, sys, torch, timm, random, numpy as np
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import RandomSampler
from matplotlib import pyplot as plt
from src.dataset import CustomDataset
from src.model import CustomModel
from tqdm import tqdm
from PIL import Image, ImageFont
from torchvision import transforms as T
from matplotlib import pyplot as plt
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
    
def get_preds(model, ts_dl, device, n_batches):   
    
    all_ims, all_preds = [], []
    
    for idx, batch in tqdm(enumerate(ts_dl)):
        if idx == n_batches: break
        ims, fts = batch
        if idx == 0: bs = ims.shape[0]
        all_ims.append(ims)
        ims, fts = ims.to(device), fts.to(device)
        preds = torch.argmax(model(ims, fts), dim = 1)
        all_preds.append(preds)
        
    return all_ims, all_preds, bs

def np2tn(tfs, np): return torch.tensor(tfs(image = np)["image"]).float().permute(2, 1, 0).unsqueeze(0)

def predict(m, path, tfs, cls_names, meta_features):
    
    fontpath = "src/SpoqaHanSansNeo-Light.ttf"
    font = ImageFont.truetype(fontpath, 200)
    im = np.array(Image.open(path))
    
    pred = torch.argmax(m(np2tn(tfs = tfs, np = im), inp_meta = meta_features), dim = 1)
    # im.save(path)
    res = cls_names[int(pred)]
    
    return im, res

def tn2np(t, t_type = "rgb"):
    
    gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return (invTrans(t) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if t_type == "gray" else (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def visualize(all_ims, all_preds, n_ims, bs, rows, cmap = None, cls_names = None, save_path = "visualization"):
    
    os.makedirs(save_path, exist_ok = True)
    assert cmap in ["rgb", "gray"], "Rasmni oq-qora yoki rangli ekanini aniqlashtirib bering!"
    if cmap == "rgb": cmap = "viridis"
    
    plt.figure(figsize = (20, 10))
    
    for idx, (ims, preds) in enumerate(zip(all_ims, all_preds)):
        if idx == n_ims: break
        rand_int = random.randint(0, (bs - 1))
        im, pred = ims[rand_int], preds[rand_int]
        # Start plot
        plt.subplot(rows, n_ims // rows, idx + 1)
        plt.imshow(tn2np(im, cmap), cmap=cmap)
        plt.axis('off')
        if cls_names is not None: plt.title(f"GT -> {cls_names[pred.item()]}")
        else: plt.title(f"Pred -> {gt}")
    
    plt.savefig(fname = f"{save_path}/results.png")
    print(f"Inference results can be found in {save_path} directory.")
    
def load_model(model_name, num_classes, checkpoint_path, url, n_features): 
    
    """
    
    This function gets several parameters and loads a classification model.
    
    Parameters:
    
        model_name      - name of a model from timm library, str;
        num_classes     - number of classes in the dataset, int;
        checkpoint_path - path to the trained model, str;
        
    Output:
    
        m               - a model with pretrained weights and in an evaluation mode, torch model object;
    
    """
    
    os.makedirs(checkpoint_path.split("/")[0], exist_ok = True)
    # Download from the checkpoint path
    if os.path.isfile(checkpoint_path): print("Pretrained model is already downloaded!"); pass
    
    # If the checkpoint does not exist
    else: 
        print("Pretrained checkpoint is not found!")
        print("Downloading the pretrained checkpoint...")
        
        # Get file id
        file_id = url.split("/")[-2]
        
        # Download the checkpoint
        os.system(f"curl -L 'https://drive.usercontent.google.com/download?id={file_id}&confirm=xxx' -o {checkpoint_path}")
    
    # Create a model based on the model name and number of classes
    m = CustomModel(model_name = model_name, n_features = n_features, n_feature_dims = [int(nd) for nd in ("512, 128").split(',')], n_cls = num_classes)
    # Load the state dictionary from the checkpoint
    m.load_state_dict(torch.load(checkpoint_path))
    print("Pretrained model is successfully loaded!")
    
    # Switch the model into evaluation mode
    return m.eval()