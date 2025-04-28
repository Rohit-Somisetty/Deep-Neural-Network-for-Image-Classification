# 🐱 Cats vs. Dogs Image Classifier 🐶

Welcome! This project tackles the classic computer vision challenge of distinguishing between images of cats and dogs using a custom-built Convolutional Neural Network (CNN) implemented in PyTorch.

It provides an end-to-end pipeline, handling dataset organization, image augmentation, model training, validation, testing, and prediction on new images **automatically**.

## ✨ Features

* **Automated Dataset Organization:** Takes the raw Kaggle "Dogs vs. Cats" dataset and automatically splits it into `train`, `validation`, and `test` sets with `cats`/`dogs` subdirectories.
* **Custom CNN Architecture:** Built from scratch using PyTorch (`torch.nn.Module`) featuring 4 convolutional layers, MaxPooling, ReLU activations, and Dropout for robust learning.
* **PyTorch Ecosystem:** Efficiently handles data loading and batching using `Dataset` and `DataLoader`.
* **Image Augmentation:** Enhances model generalization by applying random transformations (rotation, horizontal flip, color jitter) during training.
* **Standard Training Pipeline:** Includes training, validation, and testing loops with clear performance reporting (accuracy and loss).
* **Results Visualization:** Generates and saves plots of training/validation accuracy and loss curves via Matplotlib.
* **Prediction Ready:** Comes with a `predict.py` script to easily classify single images using the trained model.

## 🛠️ Tech Stack

* Python 3.x
* PyTorch
* NumPy
* Matplotlib
* Pillow (PIL)
* Kaggle API (for downloading dataset)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black)](https://matplotlib.org/)

*(See `requirements.txt` for specific versions. You can generate it using `pip freeze > requirements.txt` in your environment.)*

## 🧠 Model Architecture

The CNN processes 150x150 RGB images through the following layers:

1.  **Conv Block 1:** `Conv2d(3, 32)` -> `ReLU` -> `MaxPool2d`
2.  **Conv Block 2:** `Conv2d(32, 64)` -> `ReLU` -> `MaxPool2d`
3.  **Conv Block 3:** `Conv2d(64, 128)` -> `ReLU` -> `MaxPool2d`
4.  **Conv Block 4:** `Conv2d(128, 128)` -> `ReLU` -> `MaxPool2d`
5.  **Flatten**
6.  **Dropout (p=0.5)**
7.  **Fully Connected 1:** `Linear(6272, 512)` -> `ReLU`
8.  **Dropout (p=0.5)**
9.  **Fully Connected 2 (Output):** `Linear(512, 1)` -> `Sigmoid` (outputs probability of being a dog)

## 📊 Dataset - Kaggle Dogs vs. Cats

This project uses the **Dogs vs. Cats** dataset from the official Kaggle competition.

1.  **Download:** You'll need the Kaggle API installed (`pip install kaggle`). Download the data using the command:
    ```bash
    kaggle competitions download -c dogs-vs-cats
    ```
    This will download `train.zip`, `test1.zip`, and `sampleSubmission.csv`.

2.  **Extract:** Unzip `train.zip` directly into your project's root directory. This should create a `train/` folder containing all ~25,000 training images (`cat.####.jpg` and `dog.####.jpg`). **Do not manually create subfolders or split the data.**

