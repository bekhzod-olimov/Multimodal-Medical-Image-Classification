import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, df, mode, meta_features, transformations = None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transformations = transformations

    def __len__(self): return self.df.shape[0]

    def __getitem__(self, index):

        data = self.df.iloc[index]

        im = cv2.cvtColor(cv2.imread(data.filepath), cv2.COLOR_BGR2RGB)

        if self.transformations is not None:
            res = self.transformations(image=im)
            im = res["image"].astype(np.float32)
        else:
            im = im.astype(np.float32)

        im = im.transpose(2, 0, 1)

        if self.use_meta:
            data = (torch.tensor(im).float(), torch.tensor(self.df.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(im).float()

        if self.mode == 'test': return data
        else: return data, torch.tensor(self.df.iloc[index].target).long()
        
def get_meta_data(tr_df, test_df):
    
    tr_df['sex'] = tr_df['sex'].fillna(tr_df['sex'].mode()[0])
    tr_df['age_approx'] = tr_df['age_approx'].fillna(tr_df['age_approx'].median())
    tr_df['anatom_site_general_challenge'] = tr_df['anatom_site_general_challenge'].fillna("unknown")
    
    test_df['sex'] = test_df['sex'].fillna(test_df['sex'].mode()[0])
    test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].median())
    test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna("unknown")

    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([tr_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    tr_df = pd.concat([tr_df, dummies.iloc[:tr_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[tr_df.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    tr_df['sex'] = tr_df['sex'].map({'male': 1, 'female': 0})
    test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
    # Age features
    tr_df['age_approx'] /= 90
    test_df['age_approx'] /= 90
    
    
    # n_image per user
    tr_df['n_ims'] = tr_df.patient_id.map(tr_df.groupby(['patient_id']).image_name.count())
    test_df['n_ims'] = test_df.patient_id.map(test_df.groupby(['patient_id']).image_name.count())
    
    
    tr_df['n_ims'] = np.log1p(tr_df['n_ims'].values)
    test_df['n_ims'] = np.log1p(test_df['n_ims'].values)
    # image size
    tr_ims = tr_df['filepath'].values
    train_sizes = np.zeros(tr_ims.shape[0])
    for i, img_path in enumerate(tqdm(tr_ims)):
        train_sizes[i] = os.path.getsize(img_path)
    tr_df['im_size'] = np.log(train_sizes)
    test_images = test_df['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    test_df['im_size'] = np.log(test_sizes)

    meta_data = ['sex', 'age_approx', 'n_ims', 'im_size'] + [col for col in tr_df.columns if col.startswith('site_')]
    n_meta_data = len(meta_data)
    
    return tr_df, test_df, meta_data, n_meta_data

def get_df(data_dir, use_meta):

    tr_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    tr_df['filepath'] = tr_df['image_name'].apply(lambda x: os.path.join(data_dir, f'train', f'{x}.jpg'))
    cls_names = tr_df['benign_malignant'].unique()

    # test data
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test_df['filepath'] = test_df['image_name'].apply(lambda x: os.path.join(data_dir, f'test', f'{x}.jpg'))

    if use_meta:
        tr_df, test_df, meta_data, n_meta_data = get_meta_data(tr_df, test_df)
    else:
        meta_data = None
        n_meta_data = 0

    # class mapping
    mel_idx = 1
    
    return tr_df, test_df, meta_data, n_meta_data, mel_idx, cls_names

# df_train, df_test, meta_data, n_meta_data, mel_idx, cls_names = get_df("/mnt/data/dataset/bekhzod/im_class/skin_lesion", True)


