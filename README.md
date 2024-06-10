# Circular Deep Feature Extraction with Pretrained VGG16 and Circular DenseNet201 Alternative for AlexNet and DarkNet53

This repository contains code for extracting circular deep features from images using a combination of pretrained VGG16 and Circular DenseNet201 models. The extracted features are processed and selected using advanced statistical methods for subsequent classification tasks.

## Features

- *Circular Deep Feature Extraction*: Extracts features from circular regions within images.
- *Pre-trained Models*: Utilizes VGG16 as a substitute for AlexNet and DenseNet201 as a substitute for DarkNet53.
- *Histogram Equalization*: Enhances image contrast before feature extraction.
- *Feature Normalization*: Normalizes features to a consistent scale.
- *Feature Selection*: Uses SelectKBest and Recursive Feature Elimination (RFE) for selecting top features.
- *Classification Preparation*: Processes and prepares features for classification tasks.

## Code Overview

### Code 1: Using VGG16 (Substitute for AlexNet)

1. *Loading VGG16 Model*: Loads VGG16 model with ImageNet weights for feature extraction.
2. *Preprocessing*: Reads and preprocesses images, including resizing and histogram equalization.
3. *Circular Feature Extraction*: Extracts features from concentric circular regions within images.
4. *Normalization*: Normalizes the extracted features using MinMaxScaler.
5. *Feature Selection*: Selects top features using SelectKBest and RFE.
6. *Output Preparation*: Prepares the final dataset with selected features and corresponding labels.

### Code 2: Using DenseNet201 (Substitute for DarkNet53)

1. *Loading DenseNet201 Model*: Loads DenseNet201 model with ImageNet weights for feature extraction.
2. *Preprocessing*: Reads and preprocesses images, including resizing and histogram equalization.
3. *Circular Feature Extraction*: Extracts features from concentric circular regions within images.
4. *Normalization*: Normalizes the extracted features using MinMaxScaler.
5. *Feature Selection*: Selects top features using SelectKBest and RFE.
6. *Output Preparation*: Prepares the final dataset with selected features and corresponding labels.

## How to Use

1. *Clone the Repository*:
   sh
   git clone https://github.com/yourusername/circular-deep-feature-extraction.git
   cd circular-deep-feature-extraction
   

2. *Prepare Your Image Dataset*: Place your image files in the repository directory. Ensure the filenames contain labels as required.

3. *Install Dependencies*:
   sh
   pip install numpy opencv-python scikit-learn tensorflow keras scikit-image
   

4. *Run the Script*:
   sh
   python circular_vgg16.py  # For Code 1
   python circular_densenet201.py  # For Code 2

