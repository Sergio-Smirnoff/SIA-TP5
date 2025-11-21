import sys
from pathlib import Path

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from read_utils import plot_character, get_font3, to_binary_array
from vae import VAE
from bmp_images.bmp import BMPParser, load_bmp_dataset, save_vae_output_as_bmp, load_emojis, plot_all_emojis, draw_emoji, load_emojis_json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------
#  Dataset preparation
# ---------------------------------------------------------

def prepare_dataset():
    X = []
    for char in get_font3():
        img = to_binary_array(char)
        X.append(img.flatten())   # 35-dimensional vector

    X = np.array(X).astype(float)
    return X


# ---------------------------------------------------------
#  ASCII visualization
# ---------------------------------------------------------

def show_char(vec):
    img = vec.reshape(7, 5)
    for r in range(7):
        print("".join(["█" if x >= 0.5 else " " for x in img[r]]))


def test_with_bmp():
    parser = BMPParser()

    # dataset de imagenes bmp normalizadas [0;1]
    dataset = load_bmp_dataset("bmp_images", normalize=True)
    original_shape = dataset[0].shape

    vae = VAE(architecture=[600, 2000, 1500, 1000, 500, 200, 50, 10, 2],   # input 35 → latent 2
        learning_rate=0.005,
        optimizer="adam",
        activation_function="sigmoid",
        seed=42)

    EPOCHS = 200
    
    input_dim = dataset[0].shape[1]  # 600 * 600 * 3 = 1,080,000
    
    vae = VAE(architecture=[input_dim, 2000, 1500, 1000, 500, 200, 50, 10, 2],   # input 35 → latent 2
            learning_rate=0.005,
            optimizer="adam",
            activation_function="sigmoid",
            seed=42)


    vae.train(dataset, EPOCHS)

    for idxm, img in enumerate(dataset):
        acts_enc, zv_enc, acts_dec, zv_dec = vae.forward(img)
        print(acts_dec)
        # save_vae_output_as_bmp(acts_dec[-1][0], f"bmp_images/bmp_output/{idx}.bmp", original_shape=original_shape)


def test_with_bits():
    # La arquitectura debe empezar con 35 (tamaño de cada emoji aplanado)
    vae = VAE(architecture=[35, 25, 20, 15, 10, 5, 4],  # 35 input → 4 latent
            learning_rate=0.005,
            optimizer="adam",
            activation_function="sigmoid",
            seed=42)
    
    emojis = load_emojis_json("bmp_images/emojis.json")
    plot_all_emojis(emojis, "input_emojis.png")
    
    # Convierte a array de numpy con forma (10, 35)
    emojis_flattened = np.array([emoji.flatten() for emoji in emojis])
    
    EPOCHS = 100000
    vae.load_state_pickle(f'bmp_images/{EPOCHS}.pkl')

    # vae.train(emojis_flattened, epochs=EPOCHS)

    reconstructed_emojis = []
    for i, emoji in enumerate(emojis_flattened):
        # Encodear al espacio latente
        mu, logvar, z = vae.encode(emoji)
        
        # Decodear desde el espacio latente
        reconstructed = vae.decode(z)  # z tiene forma (1, 2)
        reconstructed_emojis.append(reconstructed[0].reshape(7, 5))
        
        print(f"Emoji {i} - Latent point (z): {z[0]}")
    plot_all_emojis(reconstructed_emojis, "reconstructed_emojis.png")

    generate_random_samples(vae)
    # vae.save_state_pickle(f'bmp_images/{EPOCHS}.pkl')


def generate_random_samples(vae):
    """Generar imágenes muestreando aleatoriamente desde el espacio latente"""
    
    n_samples = 20
    generated_images = []
    
    for i in range(n_samples):
        # Muestrear desde una distribución normal estándar
        z = np.random.randn(1, vae.latent_dim)  # (1, 2) con valores ~ N(0, 1)
        
        # Decodificar
        decoded = vae.decode(z)
        generated_images.append(decoded[0].reshape(7, 5))
        
        print(f"Sample {i} - Latent: {z[0]}")
    
    plot_all_emojis(generated_images, "random_samples.png")
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    test_with_bits()
    # print("Preparing dataset...")
    # X = prepare_dataset()

    # print("Initializing VAE...")
    # vae = VAE(
    #     architecture=[35, 16, 8, 4, 2],   # input 35 → latent 2
    #     learning_rate=0.005,
    #     optimizer="adam",
    #     activation_function="relu",
    #     seed=42
    # )

    # print("Training...")
    # epochs = 50000
    # vae.train(X, epochs)
    # print("\nTraining finished!")

    # # ---------------------------------------------------------
    # # Show reconstruction of a sample char
    # # ---------------------------------------------------------
    # idx = 5
    # print("\nOriginal:")
    # show_char(X[idx])

    # print("\nReconstruction:")
    # acts_enc, zv_enc, acts_dec, zv_dec = vae.forward(X[[idx]])
    # show_char(acts_dec[-1][0])