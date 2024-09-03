import os, sys, time, torch, pickle, argparse, pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from src.train import train_epoch, valid_epoch
from src.dataset import get_df, CustomDataset
from src.transformations import get_transformations
from src.model import CustomModel
from src.create_data import create_data
from src.utils import set_seed, plot_learning_curves, get_dls, makedirs
sys.path.append("./")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_meta', default = True)
    # parser.add_argument('--use_meta', default = False)
    parser.add_argument('--model_name', type=str, default = "rexnet_150")
    parser.add_argument('--data_dir', type=str, default='datasets/skin_lesion')
    parser.add_argument('--image_size', type=int, default = 224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--init_lr', type=float, default=3e-4)
    parser.add_argument('--n_cls', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model_dir', type=str, default='./ckpts')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_feature_dims', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args

def run(train_df, valid_df, test_df, meta_features, n_features, train_tfs, valid_tfs, mel_idx, loss_fn, seed, use_meta, threshold = 0.005, vis_path = "learning_curves", dl_save_path = "saved_dls"):

    train_dl, valid_dl, test_dl = get_dls(train_df, valid_df, test_df, meta_features, train_tfs, valid_tfs, bs = args.batch_size, nws = args.num_workers)
    makedirs([dl_save_path, vis_path])
    save_name = "with_fts" if args.use_meta else "wo_fts"
    torch.save(test_dl, f"./{dl_save_path}/{save_name}_test_dl.pth")
    print("Test dataloader is successfully saved!")

    model = CustomModel(model_name = args.model_name, n_features=n_features, n_feature_dims = [int(nd) for nd in args.n_feature_dims.split(',')], n_cls = args.n_cls).to(args.device)
    auc_max, not_improve, patience = 0., 0, 5
    best_ckpt_file  = f"{args.model_dir}/{save_name}_best.pth"
    last_ckpt_file  = f"{args.model_dir}/{save_name}_last.pth"

    optimizer = optim.Adam(model.parameters(), lr = args.init_lr)
    
    losses, accs, aucs = [], [], []
    
    for epoch in range(1, args.epochs + 1):
        print(time.ctime(), f'[Epoch {epoch}/{args.epochs}]')

        train_loss = train_epoch(model, train_dl, optimizer, use_feats = use_meta, device = args.device, image_size = args.image_size, loss_fn = loss_fn)
        val_loss, acc, auc = valid_epoch(model, valid_dl, mel_idx, use_feats = use_meta, device = args.device, n_cls = args.n_cls, loss_fn = loss_fn)
        
        announce   = "~" * 35
        free_space = " " * 9
        stats_verbose    = f"\n\n{free_space}Epoch {epoch} STATS:\n\n"
        lr_verbose       = f"Learning Rate -> {optimizer.param_groups[0]['lr']:.4f}\n"
        tr_loss_verbose  = f"Train Loss    -> {train_loss:.4f}\n"
        val_loss_verbose = f"Valid Loss    -> {val_loss:.4f}\n"
        acc_verbose      = f"Valid Acc     -> {acc:.4f}\n"
        auc_verbose      = f"Valid AUC     -> {auc:.4f}\n"
        verbose = f"\n{announce}" + stats_verbose + lr_verbose + tr_loss_verbose + val_loss_verbose + acc_verbose + auc_verbose + f"\n{announce}"
        print(verbose)
        with open(os.path.join(args.log_dir, f'log_{save_name}.txt'), 'a') as appender:
            appender.write(verbose + '\n')
        
        # Learning curves visualization
        losses.append([train_loss, val_loss]); accs.append(acc); aucs.append(auc)
        plot_learning_curves(plot_data = losses, title = "Losses", save_name = f"{save_name}_loss", save_path = vis_path, loss_data = True)
        plot_learning_curves(plot_data = accs, title = "Accuracy Score", save_name = f"{save_name}_acc", save_path = vis_path)
        plot_learning_curves(plot_data = aucs, title = "AUC Score", save_name = f"{save_name}_auc", save_path = vis_path)
        
        if auc > (auc_max + threshold):
            not_improve = 0
            print(f'Valid AUC increase [{auc_max:.4f} --> {auc:.4f}]. Saving a model with the best AUC score...')
            torch.save(model.state_dict(), best_ckpt_file)
            auc_max = auc
        else: 
            not_improve += 1
            print(f'Valid AUC + threshold ({threshold}) value did not increase for {not_improve} epochs. Current AUC score is {auc:.4f}. The best AUC score is {auc_max:.4f}')
            if not_improve == patience:
                print(f"Stopping training because validation AUC score did not improve {patience} times...")
                break
    
    torch.save(model.state_dict(), last_ckpt_file)
    

if __name__ == '__main__':

    args = parse_args()
    seed = 2024
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("saved_dls", exist_ok=True)
    set_seed(seed = seed)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    if not os.path.isdir(args.data_dir): create_data(save_dir = "datasets")
    train_df, test_df, meta_features, n_features, mel_idx, cls_names = get_df(args.data_dir, use_meta = args.use_meta)
    with open("saved_dls/cls_names", "wb") as fp: pickle.dump(cls_names, fp)
    

    train_tfs, valid_tfs = get_transformations(args.image_size)
    train_split, valid_split = train_test_split(train_df, stratify=train_df.target, test_size=0.20, random_state=42)
    train_df = pd.DataFrame(train_split); valid_df = pd.DataFrame(valid_split); test_df = pd.DataFrame(test_df)

    run(train_df = train_df, valid_df = valid_df, test_df = test_df, meta_features = meta_features, n_features = n_features, train_tfs = train_tfs, valid_tfs = valid_tfs, mel_idx = mel_idx, loss_fn = loss_fn, use_meta = args.use_meta, seed = seed)