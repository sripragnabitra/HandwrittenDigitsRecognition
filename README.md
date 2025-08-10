# Handwritten Digit Recognition — Model Comparison and Custom Image Input

## Overview
This project compares **10 different machine learning and deep learning models** for recognizing handwritten digits using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
The goal is to evaluate and compare models based on **accuracy** and **training + prediction time** (kept around ~15 seconds per model).
Then the custom handwritten digit can be given as test image and the most accurate model will predict the digit.

## Dataset
The MNIST dataset consists of:
- **60,000 training images**
- **10,000 test images**
- Grayscale, 28x28 pixels, digits 0–9.

## Preprocessing
- Data loaded from `.idx` files using custom `data_loader.py`
- Normalized pixel values to [0, 1]
- Flattened images for classical ML models
- Reshaped images to `(28, 28, 1)` for CNN-based models
- Saved preprocessed datasets in `processed/mnist_norm.pkl`

## Models Implemented
1. **Decision Tree**
2. **Random Forest**
3. **Gradient Boosting**
4. **K-Nearest Neighbors (KNN)**
5. **Logistic Regression**
6. **Support Vector Machine (SVM)**
7. **Multi-Layer Perceptron (MLP)**
8. **Convolutional Neural Network (CNN)**
9. **Deeper CNN**
10. **ResNet**

## Evaluation
- **Training + Prediction time** capped at ~15 seconds per model
- Accuracy measured on the 10,000-image MNIST test set
- Models compared based on both performance and computation time
- Altering the hyperparameters like the learning rate, pca component count, training set subset size, etc will change the model's performance.
- I have used 15sec training time as the comparing feature.
- Unlimited computational time will alter the accuracies of the models

## Obtained Accuracy Order
From lowest to highest (typical for MNIST with given constraints):
1. Decision Tree (~82-84%)
2. Gradient Boosting (~86-88%)
3. SVM (~90-92%)
4. Logistic Regression (~92–94%) 
5. Random Forest (~94–96%)
6. ResNet (~95-97%)
7. Deeper CNN (~95-97%) 
8. KNN (~96–97%) 96.86
9. CNN (~97-99%) 97.77
10. MLP (~98%+) 98.03

## Directory Structure
data/ # Raw MNIST .idx files
notebooks/ # Jupyter notebooks for each model
processed/ # Preprocessed .pkl files
results/ # Saved trained models and outputs
utils/ # Helper scripts for data loading and setup
requirements.txt # Dependencies

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the data_loader file to pre-process the data:
``` bash
python utils\data_loader.py
```

##Running the Notebooks
- Run 00_preprocessing.ipynb to know about preprocessing and data visualisation of the dataset.
- Open each model notebook (01_logistic_regression.ipynb to 10_decision_tree.ipynb) and execute the cells.
- Each notebook will: Train the model, Evaluate accuracy (acc variable), Save the accuracy in results/accuracies.csv and Save the trained model in results/<model_name>.pkl

## Testing on most accurate Model
After all models have been trained and logged:
```bash
python predict_custom_image.py --image_path "path/to/your/image.png"
```
This script will:
- Read all accuracies from results/accuracies.csv
- Identify the model with the highest accuracy
- Load the best model
- Preprocess the image
- Output the predicted digit using the best model in the terminal
  
