import os
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from skimage import exposure
import scipy.io as sio

# Load DenseNet201 model (using it as a placeholder for Darknet53)
base_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
layer1_model = Model(inputs=base_model.input, outputs=base_model.output)

# Get list of image files
files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Initialize lists to store features and labels
X = []
y = []

# Process each image file
for file in files:
    try:
        label = int(file[0])  # Assumes the file name starts with a digit representing the label
        y.append(label)

        image = cv2.imread(file)
        if image is None:
            print(f"Error reading image {file}")
            continue

        # Ensure image resizing dimensions match DenseNet201 input size
        image_resized = cv2.resize(image, (295, 295))
        image_equalized = exposure.equalize_hist(image_resized)

        center = 149  # Center of the 295x295 image
        sayac = 19
        features = []

        for i in range(8):
            size = sayac
            exm = image_equalized[center - size:center + size, center - size:center + size, :]
            exm_resized = cv2.resize(exm, (224, 224))
            exm_preprocessed = preprocess_input(exm_resized)
            exm_expanded = np.expand_dims(exm_preprocessed, axis=0)

            fm = layer1_model.predict(exm_expanded).flatten()

            features.extend(fm)
            sayac += 18

        X.append(features)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Feature selection using SelectKBest with f_classif
selector = SelectKBest(score_func=f_classif, k=1000)
X_new = selector.fit_transform(X, y)

# Use RFE for further feature selection
try:
    model = RFE(estimator=selector, n_features_to_select=1000, step=1)
    model.fit(X_new, y)
    rfe_selected = model.support_

    son1 = X_new[:, rfe_selected]
    son = np.column_stack((son1, y))
