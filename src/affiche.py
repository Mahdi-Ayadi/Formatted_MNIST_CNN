import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle

# Charger les images et les prédictions
try:
    with gzip.open("data/colored_mnist_test_images_medium.gz", 'rb') as f:
        images = pickle.load(f)
except FileNotFoundError:
    print("The dataset for the hard level is a large file that doesn't exist in this repo: make sure to download it from this link: https://drive.google.com/file/d/1L_RYPVH94ntWnm6b8okzfMobZhNr1IEh/view?usp=sharing, and add it to the data/ folder")
    raise

# Convertir les images en tableau NumPy
images = np.array(images)

predictions = np.load("predictions/predictions_medium.npy")

# Sélectionner 24 images et leurs prédictions de manière aléatoire
indices = np.random.choice(len(images), 24, replace=False)
selected_images = images[indices]
selected_predictions = predictions[indices]

# Transposer les dimensions des images pour les rendre compatibles avec imshow
selected_images = selected_images.transpose(0, 2, 3, 1)

# Créer les plots
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(selected_images[i])
    ax.set_title(f"Prediction: {selected_predictions[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
