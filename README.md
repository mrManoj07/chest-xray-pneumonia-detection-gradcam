
# Chest X-ray Pneumonia Classification with Grad-CAM

## About the Project
This project uses Deep Learning to classify chest X-ray images into two categories:
- NORMAL
- PNEUMONIA

It uses DenseNet121 with PyTorch and Grad-CAM for visual explainability.

An interactive web application is developed using Streamlit.

## Technologies Used
- Python
- PyTorch
- DenseNet121
- Grad-CAM
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn

## Project Features
- Chest X-ray image classification
- NORMAL and PNEUMONIA prediction
- Prediction confidence and class probabilities
- Grad-CAM heatmap visualization
- Interactive Streamlit web application

## Model Download
The trained model is hosted on Google Drive.

[Download Fine-tuned DenseNet121 Model](https://drive.google.com/file/d/1D39y4yAUYJ1COTmBiSJ7RcvBLucKg_ED/view?usp=sharing)

After downloading, place the model file inside the `models` folder as:

`densenet121_chest_xray_finetuned.pth`

Note: The Streamlit app can also download the model automatically from Google Drive.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/chest-xray-pneumonia-detection-gradcam.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Dataset
Chest X-ray dataset containing NORMAL and PNEUMONIA images.

## Disclaimer
This project is developed for educational and research purposes only. It is not a medical diagnostic tool. Predictions should not be used for clinical decisions.

## Author
Manoj
