# Import libraries
import os, torch, sys, pickle, argparse, numpy as np, streamlit as st
from glob import glob; from streamlit_free_text_select import st_free_text_select
from torchvision.datasets import ImageFolder
from src.utils import load_model, predict, np2tn
from src.transformations import get_transformations
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
sys.path.append("./")
st.set_page_config(layout='wide')
sys.path.append(os.getcwd())

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    # Get class names for later use
    with open(f"{args.data_path}/cls_names", "rb") as f: cls_names = pickle.load(f)
    
    save_name = "with_fts" if args.use_meta else "wo_fts"
    assert save_name in ["wo_fts", "with_fts"], "Please specify with or without features (meta data) model"
    # Get number of classes
    num_classes = len(cls_names)
    url = "https://drive.google.com/file/d/1kSyv3nXNRh1ZHcU0_-7qOlYNp4ZKk5ZE/view?usp=sharing" if "with_fts" in save_name else "https://drive.google.com/file/d/1ENyJIEf9zVu740LSxxXzVnE4NojhUl6u/view?usp=sharing"
    checkpoint_path = f"ckpts/{save_name}_best.pth"

    # Initialize transformations to be applied
    _, tfs = get_transformations(args.image_size)
    # Set a default path to the image
    default_path = glob(f"{args.root}/*.jpg")[1]

    # Load classification model
    n_features = 12 if save_name == "with_fts" else 0
    m = load_model(args.model_name, num_classes, checkpoint_path, url = url, n_features = n_features)
    st.title(f"양성/악성 AI 분류 프로그램")
    file = st.file_uploader('이미지를 업로드해주세요')
    meta_data = torch.rand(1, 12) if "with_fts" in save_name else None
    # Get image and predicted class
    im, out = predict(m = m, path = file, tfs = tfs, cls_names = cls_names, meta_features = meta_data) if file else predict(m = m, path = default_path, tfs = tfs, cls_names = cls_names, meta_features = meta_data)
    st.write(f"입력된 의료 이미지: ")
    st.image(im)

    im = tfs(image = im)["image"]

    # Initialize GradCAM object
    
    cam = GradCAM(model = m.model, target_layers = [m.model.features[-1]], use_cuda = False)

    # Get a grayscale image
    grayscale_cam = cam(np2tn(tfs = tfs, np = im).to("cpu"))[0, :]

    # Get visualization
    
    visualization = show_cam_on_image(im/255, grayscale_cam, image_weight = 0.55, colormap = 2, use_rgb = True)

    st.write(f"AI 모델 성능 확인")

    st.image(Image.fromarray(visualization))
    out = "악성" if out == "malignant" else "양성"
    st.write(f"입력된 의료 이미지는 -> {out}입니다.")
    
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Image Classification Demo")
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = "sample_ims", help = "Root folder for test images")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-sn", "--save_name", type=str, default = "with_fts")
    parser.add_argument("-um", "--use_meta", default = True)
    parser.add_argument("-is", "--image_size", type=int, default = 224)
    parser.add_argument("-dp", "--data_path", type = str, default = "saved_dls", help = "Dataset name")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 