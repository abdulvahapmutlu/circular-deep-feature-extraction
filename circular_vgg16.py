import os
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from skimage import exposure
import warnings
import scipy.io as sio

warnings.filterwarnings("ignore")

# Load a VGG16 model as a substitute for AlexNet
base_model = VGG16(weights='imagenet', include_top=True)
layer1_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
layer2_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
layer3_model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

# Get list of image files
files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize lists to store features and labels
X = []
y = []

# Process each image file
for file in files:
    label = int(file[0])
    y.append(label)

    image = cv2.imread(file)
    image_resized = cv2.resize(image, (295, 295))
    image_equalized = exposure.equalize_hist(image_resized)

    center = 149
    sayac = 19
    features = []

    for i in range(8):
        size = sayac
        exm = image_equalized[center - size:center + size, center - size:center + size, :]
        exm_resized = cv2.resize(exm, (224, 224))  # Adjusted for VGG16 input size
        exm_preprocessed = preprocess_input(exm_resized)
        exm_expanded = np.expand_dims(exm_preprocessed, axis=0)

        fm1 = layer1_model.predict(exm_expanded).flatten()
        fm2 = layer2_model.predict(exm_expanded).flatten()
        fm3 = layer3_model.predict(exm_expanded).flatten()

        features.extend(np.concatenate((fm1, fm2, fm3)))
        sayac += 18

    X.append(features)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=1000)
X_new = selector.fit_transform(X, y)

# Use RFE for further feature selection
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000)  # Use a simple model for RFE
rfe = RFE(estimator=estimator, n_features_to_select=1000, step=1)
rfe.fit(X_new, y)
rfe_selected = rfe.support_

son1 = X_new[:, rfe_selected]
son = np.column_stack((son1, y))

