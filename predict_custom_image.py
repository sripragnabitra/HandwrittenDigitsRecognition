import pandas as pd
import joblib
import numpy as np
from PIL import Image
import os
import sys

# Get image path from terminal 
if len(sys.argv) < 2:
    print("Usage: python predict_custom_image.py <image_path>")
    sys.exit(1)

IMG_PATH = sys.argv[1]
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image file not found: {IMG_PATH}")

# Find the best model 
ACC_LOG = "results/accuracies.csv"
df = pd.read_csv(ACC_LOG)

best_row = df.loc[df["accuracy_value"].idxmax()]
best_model_name = best_row["model_name"]
best_accuracy = best_row["accuracy_value"]

print(f"Best Model: {best_model_name} ({best_accuracy:.4f} accuracy)")

# Load the best model 
model_file = f"results/{best_model_name.replace('_', ' ')}.pkl"
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")

with open(model_file, "rb") as f:
    model = joblib.load(f)

#  Preprocess custom image 
img = Image.open(IMG_PATH).convert("L")  # grayscale
img = img.resize((28, 28))

img_array = np.array(img)

# Invert colors if background is black
if img_array.mean() < 127:
    img_array = 255 - img_array

# Normalize
img_array = img_array / 255.0

# Flatten or reshape depending on model type
try:
    # Classical ML models expect 1D features
    img_array = img_array.reshape(1, -1)
    prediction = model.predict(img_array)
except ValueError:
    # CNN models expect 4D input
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    prediction = np.argmax(prediction, axis=1)

# Show prediction
if prediction.ndim > 1 and prediction.shape[1] > 1:
    prediction = np.argmax(prediction, axis=1)
print(f"Predicted digit: {prediction[0]}")