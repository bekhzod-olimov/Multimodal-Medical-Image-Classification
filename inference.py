import os, sys, pickle, torch, argparse, random, numpy as np
from src.model import CustomModel
from src.utils import get_preds, visualize
from matplotlib import pyplot as plt
from torchvision import transforms as T
sys.path.append("./")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_meta', default = True)
    parser.add_argument('--model_name', type=str, default = "rexnet_150")
    parser.add_argument('--save_dir', type=str, default='visualization')
    parser.add_argument('--n_cls', type=int, default=2)
    parser.add_argument('--n_ims', default = 20)
    parser.add_argument('--model_dir', type=str, default='./ckpts')
    parser.add_argument('--vis_dir', type=str, default='./visualization')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--n_feature_dims', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)
    save_name = "with_fts" if args.use_meta else "wo_fts"
    ts_dl = torch.load("saved_dls/with_fts_test_dl.pth")
    
    with open("saved_dls/cls_names", "rb") as fp: cls_names = pickle.load(fp)
    model = CustomModel(model_name = args.model_name, n_features = 12, n_feature_dims = [int(nd) for nd in args.n_feature_dims.split(',')], n_cls = args.n_cls).to(args.device)
    model.load_state_dict(torch.load(f"{args.model_dir}/{save_name}_best.pth"))
    print("Pretrained weigts are successfully loaded!")
    
    all_ims, all_preds, bs = get_preds(model, ts_dl, args.device, n_batches = args.n_ims)
    visualize(all_ims = all_ims, all_preds = all_preds, bs = bs, n_ims = args.n_ims, rows = 4, cmap = "rgb", cls_names = cls_names, save_path = args.save_dir)
    

