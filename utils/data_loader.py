import numpy as np
import struct
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
        return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_mnist(normalise=False):
    X_train = load_mnist_images(os.path.join(DATA_DIR, "train-images.idx3-ubyte"))
    y_train = load_mnist_labels(os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
    X_test  = load_mnist_images(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"))
    y_test  = load_mnist_labels(os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))
    
    if normalise:
        X_train = X_train / 255.0
        X_test  = X_test / 255.0
    
    return X_train, y_train, X_test, y_test
