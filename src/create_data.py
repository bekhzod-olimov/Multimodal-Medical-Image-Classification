import os, sys, glob, shutil
import urllib.request as r
sys.path.append("./src")

def create_data(save_dir, data_name = "skin_lesion"):

    # Download from the checkpoint path
    if os.path.isfile(f"{save_dir}/{data_name}.csv") or os.path.isdir(f"{save_dir}/{data_name}"): print(f"The selected data is already donwloaded. Please check {save_dir}/{data_name} directory."); pass

    # If the checkpoint does not exist
    else:
        os.makedirs(save_dir, exist_ok = True)
        print("Dataset is not found!")
        print("Downloading the selected dataset...")
        url = "https://drive.google.com/file/d/1Kv54kMLthomVc2XkQ5-n8bDlYfyf1jDR/view?usp=drive_link"
        
        # Get file id
        file_id = url.split("/")[-2]
        ds_file = f"{save_dir}/{file_id}.zip"
        # Download the checkpoint
        os.system(f"curl -L 'https://drive.usercontent.google.com/download?id={file_id}&confirm=xxx' -o {ds_file}")
        # Download the dataset
        shutil.unpack_archive(ds_file, save_dir)
        os.remove(ds_file)
        os.rename(f"{save_dir}/{data_name}", f"{save_dir}/{data_name}")
        print(f"The selected dataset is downloaded and saved to {save_dir}/{data_name} directory!")

create_data(save_dir = "datasets")
