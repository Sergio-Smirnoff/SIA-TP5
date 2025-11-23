import numpy as np

from autoencoder.main import plot_latent_space
from autoenconder import BasicAutoencoder
from read_utils import get_font3, to_binary_array, plot_character
import os
import time


characters = get_font3()
X = np.array([to_binary_array(c).flatten() for c in characters])
n = X.shape[0]
labels_chars = np.array([chr(i) for i in range(0x60, 0x80)])

architecture = [35, 30, 25, 20, 15, 10, 8, 4, 2]

encoder = BasicAutoencoder(
    architecture,
    learning_rate=0.0001,
    epsilon=1e-6,
    optimizer="adam",
    activation_function="tanh",
    seed=42
)

print("Entrenando autoencoder...")
encoder.train(X, epochs=100000)
print("Entrenamiento finalizado.")

plot_latent_space(encoder, X, labels=labels_chars,
                  output_path="outputs/latent_space.png")
print("saving model state...")

np.random.seed(None)
idxA, idxB = np.random.choice(n, size=2, replace=False)
print(f"Índices seleccionados aleatoriamente: {idxA} y {idxB}")

output_dir = f"generated_letters_{int(time.time())}"
os.makedirs(output_dir, exist_ok=True)

alphas = [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.65, 0.80]

for alpha in alphas:
    new_char = encoder.generate_interpolation(X, idxA, idxB, alpha)
    img = new_char.reshape(7, 5)

    file_path = os.path.join(output_dir, f"generated_{idxA}_{idxB}_alpha{alpha:.2f}.png")
    plot_character(img, output_path=file_path)

    print(f"Generada nueva letra con alpha={alpha} → {file_path}")

origA = X[idxA].reshape(7, 5)
origB = X[idxB].reshape(7, 5)

plot_character(origA, output_path=os.path.join(output_dir, f"original_{idxA}.png"))
plot_character(origB, output_path=os.path.join(output_dir, f"original_{idxB}.png"))

print("\nProceso completado.")
print(f"Imágenes guardadas en carpeta: {output_dir}")
