import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

# ------------------ Paths ------------------
train_dir = 'images.cv_jzk6llhf18tm3k0kyttxz/data/train'
val_dir = 'images.cv_jzk6llhf18tm3k0kyttxz/data/val'
test_dir = 'images.cv_jzk6llhf18tm3k0kyttxz/data/test'
model_save_path = 'best_model.pkl'
metrics_csv_path = 'metrics.csv'
history_pkl_path = 'history.pkl'
label_txt_path = 'class_labels.txt'

# ------------------ Config ------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CNN = 20
EPOCHS_PRETRAIN = 5

# ------------------ Load Data ------------------
try:
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    val_test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    val_data = val_test_gen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    test_data = val_test_gen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    num_classes = train_data.num_classes
    class_labels = list(test_data.class_indices.keys())
    y_test_true = test_data.classes
except Exception as e:
    print(f"❌ Error in Load data:\nERROR -> {e}")


# Save class labels to file
try:
    with open(label_txt_path, 'w') as f:
        f.write('\n'.join(class_labels))
except Exception as e:
    print(f"❌ Error in Save class lables:\nERROR -> {e}")

# ------------------ Evaluation ------------------
best_model = None
best_model_name = ""
best_acc = 0
metrics_df = []
history_dict = []

def evaluate_and_save(model, name):
    try:
        global best_model, best_model_name, best_acc
        y_pred = np.argmax(model.predict(test_data), axis=1)

        acc = accuracy_score(y_test_true, y_pred)
        prec = precision_score(y_test_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test_true, y_pred)

        metrics_df.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})

        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'conf_matrix_{name}.png')
        plt.close()
    except Exception as e:
        print(f"❌ Error in evaluate_and_save() function:\nERROR -> {e}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

# ------------------ Train CNN ------------------
try:
    cnn_model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = cnn_model.fit(train_data, validation_data=val_data, epochs=EPOCHS_CNN, callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
    history_dict.append(('CNN', hist.history))
    evaluate_and_save(cnn_model, 'CNN')
except Exception as e:
    print(f"❌ Error in Train Scrached CNN model:\nERROR -> {e}")

# ------------------ Pretrained Models ------------------
pretrained_models = {
    'VGG16': VGG16,
    'ResNet50': ResNet50,
    'MobileNet': MobileNet,
    'InceptionV3': InceptionV3,
    'EfficientNetB0': EfficientNetB0
}

for name, model_fn in pretrained_models.items():
    try:
        base_model = model_fn(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit(train_data, validation_data=val_data, epochs=EPOCHS_PRETRAIN, callbacks=[EarlyStopping(patience=2)])
        history_dict.append((name, hist.history))
        evaluate_and_save(model, name)
    except Exception as e:
        print(f"❌ Error in Pre-Train model(for loop):\nERROR -> {e}")

# ------------------ Save ------------------
if best_model:
    try:
        joblib.dump(best_model, model_save_path)
    except Exception as e:
        print(f"❌ Error in if condition(best_model):\nERROR -> {e}")

pd.DataFrame(metrics_df).to_csv(metrics_csv_path, index=False)

try:
    with open(history_pkl_path, 'wb') as f:
        pickle.dump(history_dict, f)
except Exception as e:
        print(f"❌ Error in save .pkl file:\nERROR -> {e}")

