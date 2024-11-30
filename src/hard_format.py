"""This file contains the code to format the hard colored MNIST dataset."""

import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import random

# Charger les images et les prédictions
with gzip.open("data/colored_mnist_test_images_hard.gz", 'rb') as f:
    images = pickle.load(f)

# Convertir les images en tableau NumPy
images = np.array(images)
i = random.randint(0, len(images))
img = images[i]
img = img.transpose(1, 2, 0)
grayscale_img = np.mean(img, axis=-1)
# Normaliser les valeurs des pixels dans [0, 255] (if needed)
grayscale_img = (grayscale_img - grayscale_img.min()) / (grayscale_img.max() - grayscale_img.min()) * 255
grayscale_img = grayscale_img.astype(np.uint8)

binary_mask = grayscale_img > 0.5*max(grayscale_img.flatten())
if np.sum(binary_mask) > binary_mask.size / 2:
    binary_mask = ~binary_mask
formatted = np.zeros_like(grayscale_img)
formatted[binary_mask] = grayscale_img[binary_mask]

# Afficher l'image en niveaux de gris
plt.imshow(grayscale_img, cmap='gray')
plt.show()
plt.imshow(formatted, cmap='gray')
plt.show()