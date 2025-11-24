import numpy as np
import os
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from autoencoder.main import plot_latent_space
from autoenconder import BasicAutoencoder
from read_utils import get_font3, to_binary_array, plot_character


characters = get_font3()
X = np.array([to_binary_array(c).flatten() for c in characters])
n = X.shape[0]
labels_chars = np.array([chr(i) for i in range(0x60, 0x80)])

architecture = [35, 25, 18, 12, 8, 4, 2]

encoder = BasicAutoencoder(
    architecture,
    learning_rate=0.001,
    epsilon=1e-4,
    optimizer="adam",
    activation_function="tanh",
    seed=42
)

print("Entrenando autoencoder...")
encoder.train(X, epochs=100000)
print("Entrenamiento finalizado.")

plot_latent_space(
    encoder, X, labels=labels_chars,
    output_path="outputs/latent_space.png"
)

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

def plot_latent_with_interpolations(encoder, X, idxA, idxB, alphas, output_path):
    latent = encoder.get_latent_representation(X)

    A_point = latent[idxA]
    B_point = latent[idxB]

    interp_points = []
    for alpha in alphas:
        z = alpha * A_point + (1 - alpha) * B_point
        interp_points.append((alpha, z))

    plt.figure(figsize=(8, 6))

    plt.scatter(latent[:, 0], latent[:, 1], c='gray', alpha=0.4, label="Dataset")
    plt.scatter(A_point[0], A_point[1], c='red', s=120, label=f"Letra {idxA}")
    plt.scatter(B_point[0], B_point[1], c='blue', s=120, label=f"Letra {idxB}")

    for alpha, point in interp_points:
        plt.scatter(point[0], point[1], c='green', s=80)
        plt.text(point[0], point[1], f"{alpha:.2f}", fontsize=8, ha='left')

    plt.title(f"Interpolación Latente entre {idxA} y {idxB}")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Latent interpolado guardado en: {output_path}")


plot_latent_with_interpolations(
    encoder, X, idxA, idxB, alphas,
    output_path=os.path.join(output_dir, "latent_interpolations.png")
)

def build_interpolation_strip(idxA, idxB, alphas, output_dir):
    # ordenar alphas de mayor a menor
    alphas_sorted = sorted(alphas, reverse=True)

    images = []

    # A primero (alpha=1 implica A)
    origA_img = Image.open(os.path.join(output_dir, f"original_{idxA}.png"))
    images.append((f"A ({idxA})", origA_img))

    # interpolaciones descendentes
    for alpha in alphas_sorted:
        fname = f"generated_{idxA}_{idxB}_alpha{alpha:.2f}.png"
        img = Image.open(os.path.join(output_dir, fname))
        label = f"α={alpha:.2f}"
        images.append((label, img))

    # B al final (alpha=0 implica B)
    origB_img = Image.open(os.path.join(output_dir, f"original_{idxB}.png"))
    images.append((f"B ({idxB})", origB_img))

    # dimensiones
    widths, heights = zip(*(img.size for label, img in images))
    total_width = sum(widths)
    max_height = max(heights) + 40  # espacio para texto

    combined = Image.new("RGB", (total_width, max_height), "white")
    draw = ImageDraw.Draw(combined)

    x_offset = 0
    for label, img in images:
        combined.paste(img, (x_offset, 0))
        draw.text(
            (x_offset + img.width // 2 - 20, img.height + 5),
            label,
            fill="black"
        )
        x_offset += img.width

    title = f"Interpolación latente: {idxA} → {idxB}"
    draw.text((10, 10), title, fill="black")

    output_path = os.path.join(output_dir, "interpolation_strip.png")
    combined.save(output_path)

    print(f"Tira de interpolación guardada en: {output_path}")

build_interpolation_strip(idxA, idxB, alphas, output_dir)

print("\nProceso completado.")
print(f"Imágenes guardadas en carpeta: {output_dir}")
