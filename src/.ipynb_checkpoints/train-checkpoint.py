import torch, numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 

def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

def train_epoch(model, loader, optimizer, use_feats, device, image_size, loss_fn):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for idx, (data, target) in enumerate(bar):
        # if idx == 2: break
        optimizer.zero_grad()
        
        if use_feats:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, inp_meta = meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data, inp_meta = None)        
        
        loss = loss_fn(logits, target)
        loss.backward()

        if image_size in [896,576]: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description(f"Train Loss -> {loss_np:.4f} | Smooth Loss -> {smooth_loss:.4f}")

    return np.mean(train_loss)

def valid_epoch(model, loader, mel_idx, use_feats, device, n_cls, loss_fn, n_test=1, get_output=False):
    model.eval()
    val_loss, LOGITS, PROBS, TARGETS = [], [], [], []
    bar = tqdm(loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(bar):
            # if idx == 2: break
            if use_feats:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], n_cls)).to(device)
                probs = torch.zeros((data.shape[0], n_cls)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], n_cls)).to(device)
                probs = torch.zeros((data.shape[0], n_cls)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = loss_fn(logits, target)
            loss_np = loss.detach().cpu().numpy()
            val_loss.append(loss_np)
            smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
            bar.set_description(f"Valid Loss -> {loss_np:.4f} | Smooth Loss -> {smooth_loss:.4f}")

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    
    if get_output: return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() 
        auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        return val_loss, acc, auc
    
