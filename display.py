import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle

# ------------------ Paths ------------------
model_path = 'best_model.pkl'
metrics_path = 'metrics.csv'
history_path = 'history.pkl'
label_txt_path = 'class_labels.txt'
conf_matrix_files = [
    'conf_matrix_CNN.png',
    'conf_matrix_VGG16.png',
    'conf_matrix_ResNet50.png',
    'conf_matrix_MobileNet.png',
    'conf_matrix_InceptionV3.png',
    'conf_matrix_EfficientNetB0.png'
]

# ------------------ UI Config ------------------

st.set_page_config(page_title="🐟 Fish Classifier Dashboard", layout="centered")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        font-weight: bold;
    }
    .header-main {
        text-align: center;
        font-size: 48px;
        color: #EC7063;
        font-weight: 900;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 30px;
        color: #2ECC71;
        margin-top: 40px;
    }
    .conf-title {
        font-size: 24px;
        color: #5D6D7E;
        margin-bottom: 10px;
    }
    .prediction-label {
        font-size: 26px;
        color: #1F618D;
        font-weight: 700;
    }
    .confidence-label {
        font-size: 26px;
        color: #B03A2E;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-main'>🐟 Fish Image Classifier</div>", unsafe_allow_html=True)

# ------------------ Load Labels ------------------
if os.path.exists(label_txt_path):
    with open(label_txt_path, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]
else:
    class_labels = []

# ------------------ Show Metrics ------------------
if os.path.exists(metrics_path):
    st.markdown("<div class='section-title'>📊 Model Comparison Metrics</div>", unsafe_allow_html=True)
    metrics_df = pd.read_csv(metrics_path)
    melted_df = pd.melt(metrics_df, id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x='Model', y='Score', hue='Metric', palette="Set2")
    plt.title("Accuracy, Precision, Recall, F1-Score")
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    st.pyplot(plt.gcf())
    plt.close()

# ------------------ Plot History ------------------
if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history_dict = pickle.load(f)

    def plot_metric(metric):
        plt.figure()
        for name, hist in history_dict:
            plt.plot(hist[metric], label=f'{name} Train', linestyle='-')
            plt.plot(hist[f'val_{metric}'], label=f'{name} Val', linestyle='--')
        plt.title(f"Model {metric.capitalize()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

    st.markdown("<div class='section-title'>📈 Training History</div>", unsafe_allow_html=True)
    plot_metric('accuracy')
    plot_metric('loss')

# ------------------ Confusion Matrices ------------------
st.markdown("<div class='section-title'>🧩 Confusion Matrices</div>", unsafe_allow_html=True)
for file in conf_matrix_files:
    if os.path.exists(file):
        model_name = file.replace("conf_matrix_", "").replace(".png", "")
        st.markdown(f"<div class='conf-title'>📌 {model_name} Model</div>", unsafe_allow_html=True)
        st.image(file, caption=f"{model_name}", use_container_width=True)

# ------------------ Image Upload and Prediction ------------------
st.markdown("<div class='section-title'>🔍 Upload a Fish Image</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        preds = model.predict(img_array)[0]
        top_idx = np.argmax(preds)
        confidence = preds[top_idx]

        class_name = class_labels[top_idx] if class_labels else f"Class {top_idx}"

        st.markdown(f"<div class='prediction-label'>🎯 Predicted Class: <strong>{class_name}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence-label'>🔬 Confidence Score: <strong>{confidence * 100:.2f}%</strong></div>", unsafe_allow_html=True)
    else:
        st.error("❌ Model file not found.")
