# 🐱🐶 Image Classification (CNN): Cats vs Dogs

A Convolutional Neural Network (CNN)-based binary image classification project to distinguish between images of cats and dogs.  
Implemented using TensorFlow/Keras, with data augmentation techniques for improving model generalization.

---

## 📚 Project Overview

- **Objective**: Train a CNN to classify input images as either a **cat** or a **dog**.
- **Dataset**: Kaggle’s Dogs vs. Cats dataset (12,000+ images).
- **Model Features**:
  - Sequential CNN architecture
  - MaxPooling layers for down-sampling
  - Data augmentation (rotation, shift, shear, zoom, flip)
  - Binary cross-entropy loss function
  - RMSprop optimizer

---

## 📂 Repository Structure

| File / Folder | Purpose |
|---------------|---------|
| `Assignment 6.ipynb` | Jupyter Notebook containing the full training pipeline |
| `cats_and_dogs_small_1.h5` | Saved trained model (basic CNN without heavy augmentation) |
| `cats_and_dogs_small_2.h5` | Saved trained model (CNN with data augmentation) |
| `train/`, `validation/`, `test/` | (Optional) Organized dataset folders for training/validation/testing |
| `README.md` | Project documentation |

---

