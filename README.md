# Multimodal Medical Image Classification using Deep Learning
This repository contains a deep learning (DL)-based artificial intelligence (AI) multi-modal model to classify medical images not only using images but also extra features (meta-data). 

# These are the steps to use this repository:

1. Clone the repository:

`git clone https://github.com/bekhzod-olimov/Multimodal-Medical-Image-Classification.git`

`cd Multimodal-Medical-Image-Classification`

2. Create conda environment and activate it using the following script:
   
`conda create -n ENV_NAME python=3.10`

`conda activate ENV_NAME`

(if necessary) add the created virtual environment to the jupyter notebook (helps with debugging)

`python -m ipykernel install --user --name ENV_NAME --display-name ENV_NAME`

3. Install dependencies using the following script:

`pip install -r requirements.txt`

4. Train the models:

The training data is automatically downloaded and saved to the datasets directory once the training script is run

`python main.py`

4. Run demo using the following script:

`streamlit run demo.py`

Please note that it will take considerable amount of time when the script is run for the first time. Because all tasks in the system (segmentation, detection, colorization) pretrained AI models' weights need to be downloaded.
