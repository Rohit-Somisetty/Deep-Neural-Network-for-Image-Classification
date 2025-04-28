# 🐱 Cats vs. Dogs Image Classifier 🐶

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images as containing either a cat or a dog. It uses pre-split training, validation, and test datasets to train and evaluate the model.

## ✨ Features

* **Custom CNN Architecture:** Built from scratch using PyTorch (`torch.nn.Module`) with 4 convolutional layers, MaxPooling, ReLU activations, and Dropout for regularization.
* **PyTorch Ecosystem:** Leverages `Dataset` and `DataLoader` for efficient data loading and batching from pre-structured folders.
* **Image Augmentation:** Applies random transformations (rotation, horizontal flip, color jitter) to the training data to improve model generalization.
* **Training & Validation:** Includes standard training and validation loops to monitor performance during training using separate datasets.
* **Results Visualization:** Plots and saves training/validation accuracy and loss curves using Matplotlib.
* **Prediction Script:** Provides a separate script (`predict.py`) to load the trained model and classify a single image.

## 🛠️ Tech Stack

* Python 3.x
* PyTorch
* NumPy
* Matplotlib
* Pillow (PIL)

*(See `requirements.txt` for specific versions)*

## 🧠 Model Architecture

The CNN consists of the following main components:

1.  **Input:** 150x150 RGB images (3 channels)
2.  **Convolutional Blocks:** Four blocks of `Conv2d` -> `ReLU` -> `MaxPool2d`. The number of filters increases (32 -> 64 -> 128 -> 128).
3.  **Flattening:** Flattens the output of the last pooling layer.
4.  **Dropout:** Applies dropout (p=0.5) for regularization.
5.  **Fully Connected Layers:** Two `Linear` layers (first with ReLU, second outputting a single value).
6.  **Output Activation:** `Sigmoid` function to produce a probability score between 0 (cat) and 1 (dog).


## 📊 Dataset

This project uses the **Dogs vs. Cats** dataset from the Kaggle competition.

1.  **Download:** You'll need the Kaggle API installed (`pip install kaggle`). Download the data using the command:
    ```bash
    kaggle competitions download -c dogs-vs-cats
    ```
    This will download `train.zip`, `test1.zip`, and `sampleSubmission.csv`.

2.  **Extract:** Unzip `train.zip`. This will create a `train/` folder containing ~25,000 `.jpg` images named like `cat.####.jpg` and `dog.####.jpg`.

3.  **Manual Splitting Required:** The provided Python scripts expect the data to be **pre-split** into `train`, `validation`, and `test` directories, each containing `cats` and `dogs` subdirectories. **You must create this structure manually** from the images extracted from `train.zip`.

    * Create the following directories in your project root:
        * `train/cats/`
        * `train/dogs/`
        * `validation/cats/`
        * `validation/dogs/`
        * `test/cats/`
        * `test/dogs/`
    * Decide on your split ratio (e.g., 80% train, 10% validation, 10% test, or similar to the original script's 70/15/15).
    * Go through the unzipped `train/` folder from Kaggle.
    * Copy the appropriate number of `cat.####.jpg` images into `train/cats/`, `validation/cats/`, and `test/cats/`.
    * Copy the appropriate number of `dog.####.jpg` images into `train/dogs/`, `validation/dogs/`, and `test/dogs/`.

