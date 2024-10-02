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

4. Train the model:

a) download the dataset for training: the training data is automatically downloaded and saved to the datasets (by default) directory using this script:

`python src/create_data.py`

Train process arguments can be changed based on the following information:

![image](https://github.com/user-attachments/assets/38fea799-ef4d-46c7-941b-208f02d97f5a)

`python main.py`

5. Inference using the pretrained model:

`python inference.py`

Inference process arguments can be changed based on the following information:

![image](https://github.com/user-attachments/assets/2265ca08-90e2-48b0-a722-f0181a4685b2)

6. Run demo using the following script:

`streamlit run demo.py`
