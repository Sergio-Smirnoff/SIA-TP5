from autoenconder import BasicAutoencoder
from main import plot_latent_space
from vae_main import prepare_emojis_for_plot
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
    gae = BasicAutoencoder(architecture=[input_dim, 2000, 50, 2],
                           learning_rate=0.001,
                           optimizer="adam",
                           activation_function="tanh",
                           seed=None)

    EPOCHS = 2000
    print("\nEntrenando GAE...")
    # gae.train(dataset_flat, epochs=EPOCHS)
    gae.initialize_from_file_pickle(f'outputs/bmp_images/gae_bmp_tanh.pkl')
    plot_latent_space(
        encoder=gae,
        X=dataset_flat,
        labels=[f'BMP: {label}' for label in range(len(dataset_flat))],
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

    # for z in latent_codes:
    #     z_mod = z.copy()

    #     z_mod[0, 0] += 1.0

    #     decoded = gae.decode_from_latent(z_mod)
    #     new_images.append(decoded)

    z_mod_0 = latent_codes[0].copy()
    z_mod_0[0, 0] += -0.2  # sumar 1 en X
    z_mod_0[0, 1] += 0.2 # sumar 1 en X
    new_emoji = gae.decode_from_latent(z_mod_0)
    new_images.append(new_emoji[0])  

    z_mod_0 = latent_codes[0].copy()
    z_mod_0[0, 0] += -0.4  # sumar 1 en X
    z_mod_0[0, 1] += 0.4 # sumar 1 en X
    new_emoji = gae.decode_from_latent(z_mod_0)
    new_images.append(new_emoji[0]) 

    z_mod_0 = latent_codes[0].copy()
    z_mod_0[0, 0] += -0.8  # sumar 1 en X
    z_mod_0[0, 1] += 0.8 # sumar 1 en X
    new_emoji = gae.decode_from_latent(z_mod_0)
    new_images.append(new_emoji[0])  

    z_mod_0 = latent_codes[2].copy()
    z_mod_0[0, 0] += 100  # sumar 1 en X
    # z_mod_0[0, 1] += 1 # sumar 1 en X
    new_emoji = gae.decode_from_latent(z_mod_0)
    new_images.append(new_emoji[0])   

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

    gae.save_state_pickle(f'bmp_images/gae_bmp_tanh.pkl')
    print("\n===== TEST COMPLETADO EXITOSAMENTE =====")


def test_with_bits():
    gae = BasicAutoencoder(architecture=[35, 25, 15, 2],
              learning_rate=0.005,
              optimizer="adam",
              activation_function="tanh",
              seed=None)

    emojis = load_emojis_json("bmp_images/emojis.json")
    plot_all_emojis(emojis, "input_emojis.png")

    emojis_flattened = np.array([emoji.flatten() for emoji in emojis])

    EPOCHS = 100000
    gae.initialize_from_file_pickle(f'outputs/bmp_images/gaebits.pkl')
    
    print("EMOJIS")
    print(emojis_flattened)
    # history = gae.train(emojis_flattened, epochs=EPOCHS)

    plot_latent_space(
        encoder=gae, 
        X=emojis_flattened,
        labels=[f'Emoji: {label}' for label in range(len(emojis_flattened))],
        output_path="bmp_images/gae_bit_output/latent_space_gae.png"
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
    
# for z in new_zs:
    z_mod_0 = new_zs[4].copy()
    z_mod_0[0, 1] += -0.75  # sumar 1 en Y
    z_mod_0[0, 0] += 0.75  # sumar 1 en X
    new_emoji = gae.decode_from_latent(z_mod_0)
    new_emojis.append(new_emoji[0])  # <-- vector 35# for z in new_zs:

    z_mod_0 = new_zs[4].copy()
    z_mod_0[0, 1] += -1.25  # sumar 1 en Y
    z_mod_0[0, 0] += 1.25  # sumar 1 en X
    new_emoji = gae.decode_from_latent(z_mod_0)
    new_emojis.append(new_emoji[0])  # <-- vector 35

    # for z in new_zs:
    z_mod_0 = new_zs[4].copy()
    z_mod_0[0, 1] += -2  # sumar 1 en Y
    z_mod_0[0, 0] += 2  # sumar 1 en X
    new_emoji = gae.decode_from_latent(z_mod_0)
    new_emojis.append(new_emoji[0])  # <-- vector 35

    reconstructed_bin = prepare_emojis_for_plot(reconstructed_emojis)
    new_bin = prepare_emojis_for_plot(new_emojis)

    plot_all_emojis(reconstructed_bin, "reconstructed_emojis_gae.png")
    plot_all_emojis(new_bin, "new_emojis_gae.png")

    gae.save_state_pickle(f'bmp_images/gaebits.pkl')


if __name__ == "__main__":
    # test_with_bits()
    test_with_bmp_gae()
