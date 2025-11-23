import json
import numpy as np
import os
import time

from autoenconder import BasicAutoencoder
from read_utils import plot_character

JSON_PATH = "emoji.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

emojis = data["emojis"]

X = np.array([np.array(e).flatten() for e in emojis])
n = X.shape[0]

print(f"Dataset cargado desde {JSON_PATH}")
print(f"Total de emojis: {n}, tamaño por emoji: {X.shape[1]} pixeles")

architecture = [35, 30, 25, 20, 15, 10, 8, 4, 2]

encoder = BasicAutoencoder(
    architecture,
    learning_rate=0.0001,
    epsilon=1e-6,
    optimizer="adam",
    activation_function="tanh",
    seed=None
)

print("Entrenando autoencoder...")
encoder.train(X, epochs=25000)
print("Entrenamiento finalizado.")


np.random.seed(None)
idxA, idxB = np.random.choice(n, size=2, replace=False)

print(f"Índices seleccionados aleatoriamente: {idxA} y {idxB}")

output_dir = f"generated_emojis_{int(time.time())}"
os.makedirs(output_dir, exist_ok=True)

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for alpha in alphas:
    new_char = encoder.generate_interpolation(X, idxA, idxB, alpha)
    img = new_char.reshape(7, 5)

    file_path = os.path.join(output_dir, f"generated_{idxA}_{idxB}_alpha{alpha:.2f}.png")
    plot_character(img, output_path=file_path)

    print(f"Emoji generado (alpha={alpha:.2f}) → {file_path}")

origA = X[idxA].reshape(7, 5)
origB = X[idxB].reshape(7, 5)

plot_character(origA, output_path=os.path.join(output_dir, f"original_{idxA}.png"))
plot_character(origB, output_path=os.path.join(output_dir, f"original_{idxB}.png"))

print("\nProceso completado.")
print(f"Imágenes guardadas en carpeta: {output_dir}")
