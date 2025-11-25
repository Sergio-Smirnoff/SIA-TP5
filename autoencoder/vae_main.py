import sys
from pathlib import Path
import time
# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from read_utils import plot_character, get_font3, to_binary_array
from vae import VAE
from bmp_images.bmp import BMPParser, load_bmp_dataset, save_vae_output_as_bmp, load_emojis, plot_all_emojis, draw_emoji, load_emojis_json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_with_bmp_vae():
    print("\n======= TEST VAE WITH BMP IMAGES =======")

    # -----------------------------
    # 1. Leer dataset BMP
    # -----------------------------
    print("Cargando dataset BMP...")
    dataset = load_bmp_dataset("bmp_images", normalize=True)

    n_images, H, W, C = dataset.shape
    input_dim = H * W * C

    print(f"Dataset shape: {dataset.shape}  -> input_dim={input_dim}")

    # Flatten para VAE: (n_images, input_dim)
    dataset_flat = dataset.reshape(n_images, input_dim)

    # -----------------------------
    # 2. Crear el VAE
    # -----------------------------
    vae = VAE(
        architecture=[
            input_dim,
            2000,
            500,
            2
        ],
        learning_rate=0.001,
        optimizer="adam",
        activation_function="tanh",
        seed=42
    )

    # -----------------------------
    # 3. Entrenar
    # -----------------------------
    EPOCHS = 300
    print("\nEntrenando VAE...")
    # vae.train(dataset_flat, epochs=EPOCHS)
    vae.load_state_pickle("bmp_images/usame.pkl")

    # -----------------------------
    # 4. Reconstruir las imágenes
    # -----------------------------
    reconstructed = []
    latent_codes = []

    print("\nReconstruyendo imágenes...")

    for img_flat in dataset_flat:
        mu, logvar, z = vae.encode(img_flat)
        z = np.array(z).reshape(1, -1)

        decoded = vae.decode(z)  # (1, input_dim)
        reconstructed.append(decoded[0])
        latent_codes.append(z.copy())

    reconstructed = np.array(reconstructed)  # (n_images, input_dim)
    reconstructed_imgs = reconstructed.reshape(n_images, H, W, C)

    # -----------------------------
    # 5. Generar nuevas imágenes moviendo el espacio latente
    # -----------------------------
    print("\nGenerando nuevas imágenes desde el espacio latente...")
    new_images = []

    # for z in latent_codes:
    z_mod_0 = latent_codes[0].copy()
    z_mod_0[0, 0] += 1.0
    # z_mod[0, 1] += -2.0
    decoded = vae.decode(z_mod_0)
    new_images.append(decoded[0])

    # for z in latent_codes:
    z_mod_0 = latent_codes[0].copy()
    z_mod_0[0, 0] += 2.0
    # z_mod[0, 1] += -2.0
    decoded = vae.decode(z_mod_0)
    new_images.append(decoded[0])

    # for z in latent_codes:
    z_mod_0 = latent_codes[0].copy()
    z_mod_0[0, 0] += 3.0
    # z_mod[0, 1] += -2.0
    decoded = vae.decode(z_mod_0)
    new_images.append(decoded[0])

    # for z in latent_codes:
    z_mod_0 = latent_codes[0].copy()
    z_mod_0[0, 0] += 4.0
    # z_mod[0, 1] += -2.0
    decoded = vae.decode(z_mod_0)
    new_images.append(decoded[0])
    


    new_images = np.array(new_images).reshape(n_images, H, W, C)

    # -----------------------------
    # 6. Guardar imágenes reconstruidas y nuevas
    # -----------------------------
    out_dir_rec = Path("bmp_images/bmp_output/reconstructed")
    out_dir_new = Path("bmp_images/bmp_output/generated")
    out_dir_rec.mkdir(parents=True, exist_ok=True)
    out_dir_new.mkdir(parents=True, exist_ok=True)

    print("\nGuardando imágenes BMP...")

    for i in range(n_images):
        print(f'intput: \n{dataset_flat[i]}')
        print(f'output:\n{reconstructed_imgs[i]}')

        save_vae_output_as_bmp(
            reconstructed_imgs[i],
            str(out_dir_rec / f"reconstructed_{i}.bmp"),
            original_shape=(H, W, C)
        )

        save_vae_output_as_bmp(
            new_images[i],
            str(out_dir_new / f"generated_{i}.bmp"),
            original_shape=(H, W, C)
        )

    # vae.save_state_pickle(f'usame.pkl')
    print("\n===== TEST COMPLETADO EXITOSAMENTE =====")




def test_with_bits():
    # La arquitectura debe empezar con 35 (tamaño de cada emoji aplanado)
    vae = VAE(architecture=[35,25,15,2],  # 35 input → 4 latent
            learning_rate=0.005,
            optimizer="adam",
            activation_function="sigmoid",
            seed=42)
    
    emojis = load_emojis_json("bmp_images/emojis.json")
    plot_all_emojis(emojis, "input_emojis.png")
    
    # Convierte a array de numpy con forma (10, 35)
    emojis_flattened = np.array([emoji.flatten() for emoji in emojis])
    
    EPOCHS = 10000
    # vae.load_state_pickle(f'bmp_images/10000_191657-24-11-2025.pkl')

    print("EMOJIS")
    print(emojis_flattened)
    history = vae.train(emojis_flattened, epochs=EPOCHS)

    activations_enc, z_values_enc, activations_dec, z_values_dec = vae.forward(emojis_flattened)
    print('RESULT EMOJIS')
    print(activations_dec[-1])

    reconstructed_emojis = []
    new_emojis = []
    new_zs = []

    for i, emoji in enumerate(emojis_flattened):

        mu, logvar, z = vae.encode(emoji)

        # Asegurar shape (1,2)
        z = np.array(z).reshape(1, -1)

        # Reconstrucción normal
        reconstructed = vae.decode(z)
        reconstructed_emojis.append(reconstructed[0])  # <-- 35 valores

        # Guardamos el z original
        new_zs.append(z.copy())

        print(f"Emoji {i} - Latent point (z): {z}")

    # ---- Generación de nuevos samples ----

    # for z in new_zs:
    z_mod_0 = new_zs[0].copy()
    z_mod_0[0, 1] += -0.5  # sumar 1 en Y
    z_mod_0[0, 0] += -0.5  # sumar 1 en X
    new_emoji = vae.decode(z_mod_0)
    new_emojis.append(new_emoji[0])  # <-- vector 35

    z_mod_0 = new_zs[0].copy()
    z_mod_0[0, 1] += -0.3  # sumar 1 en Y
    z_mod_0[0, 0] += +1  # sumar 1 en X
    new_emoji = vae.decode(z_mod_0)
    new_emojis.append(new_emoji[0])  # <-- vector 35

    # ---- Plot ----

    reconstructed_bin = prepare_emojis_for_plot(reconstructed_emojis)
    new_bin = prepare_emojis_for_plot(new_emojis)

    plot_all_emojis(reconstructed_bin, "reconstructed_emojis.png")
    plot_all_emojis(new_bin, "new_emojis_XY.png")

    # vae.save_state_pickle(f'bmp_images/{EPOCHS}_{time.strftime("%H%M%S-%d-%m-%Y")}.pkl')

def prepare_emojis_for_plot(decoded_output, threshold=0.5):
    """
    Normaliza y reformatea la salida del decoder para plot_all_emojis
    
    Args:
        decoded_output: Salida del vae.decode() - shape (n_emojis, 35) con valores [0,1]
        threshold: Umbral para binarizar (default 0.5)
    
    Returns:
        Array con shape (n_emojis, 7, 5) con valores binarios 0 o 1
    """
    if not isinstance(decoded_output, np.ndarray):
        decoded_output = np.array(decoded_output)
    # Binarizar: valores > threshold → 1, sino → 0
    binary = (decoded_output > threshold).astype(int)
    
    # Reshape de (n_emojis, 35) a (n_emojis, 7, 5)
    reshaped = binary.reshape(-1, 7, 5)
    
    return reshaped

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

    # test_with_bits()
    test_with_bmp_vae()
    