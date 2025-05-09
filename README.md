# 🐟 Fish Image Classifier

## 🧠 Problem Statement
This project focuses on classifying fish images into multiple categories using deep learning. The goal is to build a robust image classification pipeline that uses both a custom CNN and transfer learning techniques, and deploy it as a web app for user-friendly predictions.

---

## 📌 Problem Description
Fish classification can be a vital task in aquaculture, marine biology, and commercial fishing. This project automates the classification process by:

- Training a **Convolutional Neural Network (CNN)** from scratch.
- Using **pretrained models** like VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0 for transfer learning.
- Evaluating models based on **accuracy, precision, recall, F1-score**, and **confusion matrices**.
- Saving the best-performing model.
- Creating an interactive **Streamlit** web app that allows users to upload images and get predictions.

---

## 🔄 Project Flow

1. **Data Preparation**:
   - Data is split into `train`, `val`, and `test` folders under `images.cv_jzk6llhf18tm3k0kyttxz/data/`.

2. **Model Training** (`cnn.py`):
   - A scratch CNN model is built and trained.
   - Pretrained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0) are fine-tuned.
   - Each model's performance is evaluated on test data.
   - The best model is saved as `best_model.pkl`.
   - Metrics, class labels, training history, and confusion matrices are saved.

3. **Web App Deployment** (`display.py`):
   - Streamlit-based app loads saved metrics and visualizations.
   - Allows users to upload images and predicts the fish class using the best saved model.
   - Displays confidence scores and prediction result.

---

## 📦 Python Packages Used

- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `pickle`
- `Pillow`
- `streamlit`

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fish-image-classifier.git
   cd fish-image-classifier
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, use:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn joblib pillow streamlit
   ```

4. To train the model (ensure the image folders are correctly placed):
   ```bash
   python cnn.py
   ```

5. To launch the web app:
   ```bash
   streamlit run display.py
   ```

---

## 📁 Project Structure

```
fish-image-classifier/
│
├── cnn.py                  # Train and evaluate CNN and pretrained models
├── display.py              # Streamlit app for prediction and visualizations
├── class_labels.txt        # Saved class names
├── best_model.pkl          # Best performing model
├── metrics.csv             # Model evaluation metrics
├── history.pkl             # Training history
├── conf_matrix_*.png       # Confusion matrices
└── images.cv_jzk6llhf18tm3k0kyttxz/   # Dataset folder with train/val/test
```

---

## ⚖️ License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share it with proper attribution.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## 🙌 Acknowledgments

- Pretrained models from TensorFlow Keras Applications.
- Streamlit for rapid UI development.
- Matplotlib & Seaborn for visualizations.

---

