from autoencoder.autoenconder import BasicAutoencoder
from autoencoder.main import plot_latent_space
from autoencoder.vae_main import prepare_emojis_for_plot
from bmp_images.bmp import load_bmp_dataset, save_vae_output_as_bmp, load_emojis_json, plot_all_emojis
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))


def test_with_bmp_gae():
    print("\n======= TEST GAE WITH BMP IMAGES =======")

    print("Cargando dataset BMP...")
    dataset = load_bmp_dataset("bmp_images", normalize=True)

    n_images, H, W, C = dataset.shape
    input_dim = H * W * C

    print(f"Dataset shape: {dataset.shape}  -> input_dim={input_dim}")

    dataset_flat = dataset.reshape(n_images, input_dim)
    gae = BasicAutoencoder(architecture=[input_dim, 2000, 1000, 500, 250, 100, 50, 2],
                           learning_rate=0.001,
                           optimizer="adam",
                           activation_function="sigmoid",
                           seed=None)

    EPOCHS = 500
    print("\nEntrenando GAE...")
    gae.train(dataset_flat, epochs=EPOCHS)
    # vae.load_state_pickle("bmp_images/500_200621-23-11-2025.pkl")  el q daba bien el ryu
    plot_latent_space(
        gae, dataset_flat,
        output_path="bmp_images/bmp_output/latent_space.png"
    )

    reconstructed = []
    latent_codes = []

    print("\nReconstruyendo imágenes...")

    for img_flat in dataset_flat:
        latent = gae.get_latent_representation(img_flat)
        latent = np.array(latent).reshape(1, -1)

        decoded = gae.decode_from_latent(latent)  # (1, input_dim)
        reconstructed.append(decoded)
        latent_codes.append(latent.copy())

    reconstructed = np.array(reconstructed)  # (n_images, input_dim)
    reconstructed_imgs = reconstructed.reshape(n_images, H, W, C)

    print("\nGenerando nuevas imágenes desde el espacio latente...")
    new_images = []

    for z in latent_codes:
        z_mod = z.copy()

        z_mod[0, 0] += 1.0

        decoded = gae.decode_from_latent(z_mod)
        new_images.append(decoded)

    new_images = np.array(new_images).reshape(n_images, H, W, C)

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

    #vae.save_state_pickle(f'bmp_images/{EPOCHS}_{time.strftime("%H%M%S-%d-%m-%Y")}.pkl')
    print("\n===== TEST COMPLETADO EXITOSAMENTE =====")


def test_with_bits():
    gae = BasicAutoencoder(architecture=[35, 25, 15, 2],
              learning_rate=0.005,
              optimizer="adam",
              activation_function="sigmoid",
              seed=None)

    emojis = load_emojis_json("bmp_images/emojis.json")
    plot_all_emojis(emojis, "input_emojis.png")

    emojis_flattened = np.array([emoji.flatten() for emoji in emojis])

    EPOCHS = 100000
    # vae.load_state_pickle(f'bmp_images/10000_143709-23-11-2025.pkl')

    print("EMOJIS")
    print(emojis_flattened)
    history = gae.train(emojis_flattened, epochs=EPOCHS)

    plot_latent_space(
        gae, emojis_flattened,
        output_path="bmp_images/bit_output/latent_space.png"
    )

    activations, z_values = gae.forward(emojis_flattened)
    print('RESULT EMOJIS')
    print(activations[-1])

    reconstructed_emojis = []
    new_emojis = []
    new_zs = []

    for i, emoji in enumerate(emojis_flattened):
        latent = gae.get_latent_representation(emoji)
        latent = np.array(latent).reshape(1, -1)
        reconstructed = gae.decode_from_latent(latent)
        reconstructed_emojis.append(reconstructed)

        new_zs.append(latent)

        print(f"Emoji {i}")


    for z in new_zs:
        z_mod = z.copy()

        print(f'ZPREV={z_mod}')
        z_mod[0, 1] += 1.0  # sumar 1 en Y
        # z_mod[0, 0] += 0.3  # sumar 1 en Y

        print(f'ZMOD={z_mod}')
        new_emoji = gae.decode_from_latent(z_mod)
        new_emojis.append(new_emoji[0])

    reconstructed_bin = prepare_emojis_for_plot(reconstructed_emojis)
    new_bin = prepare_emojis_for_plot(new_emojis)

    plot_all_emojis(reconstructed_bin, "reconstructed_emojis.png")
    plot_all_emojis(new_bin, "new_emojis_XY.png")

if __name__ == "__main__":
    # test_with_bits()
    test_with_bmp_gae()
