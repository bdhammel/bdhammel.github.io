import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mvn
import os
from keras.datasets import mnist as mnist_loader


class MNIST:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()
        self.target = y_test
        self.data = x_test.reshape(len(x_test), -1)

mnist = MNIST()

DATASET_DIR = './datasets'

if not os.path.exists(DATASET_DIR):
    print("Creating a directory for example datasets")
    os.makedirs(DATASET_DIR)

#mnist = fetch_mldata('MNIST original', data_home=DATASET_DIR)
digit_values = np.unique(mnist.target)

digit_params = {}
for digit_value in digit_values:
    digits = mnist.data[mnist.target == digit_value]
    digit_params[digit_value] = {
                "mean": digits.mean(axis=0),
                "std": digits.std(axis=0)
    }

intensities = np.arange(0,255)

fig, axs = plt.subplots(nrows=10, figsize=(8,11))
axs[0].set_title("Probability Maps")

for digit_value in digit_values:
    
    prob_map = np.array([ 
        np.exp(-(intensities - mu)**2 / (1e-3 + 2 * (sig)**2))
        for mu, sig in zip(digit_params[digit_value]['mean'], digit_params[digit_value]['std'])
    ])  
    
    ax = axs[digit_value]
    im = ax.imshow(prob_map.T, origin='lower')
    ax.set_aspect('auto')
    ax.set_ylabel(f"{digit_value:.0f}", rotation='horizontal')
    ax.set_yticks([0,255])
    if digit_value != 9:
        ax.set_xticks([])
    else:
        ax.set_xlabel("Pixel Number")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)

ax.set_xlabel("pixel location")
fig.text(0.01, 0.5, "Pixel value", va='center', rotation='vertical')
fig.text(0.82, 0.5, "Probability", va='center', rotation='vertical')
plt.show()
