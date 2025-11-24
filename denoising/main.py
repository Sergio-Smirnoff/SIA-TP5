import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from error_utils import compare_pixel_error, plot_error, read_errors_from_file
from autoenconder import BasicAutoencoder, add_salt_and_pepper_noise, add_gaussian_noise
import logging as log
from read_utils import plot_character, get_font3, to_binary_array
from arch_comp import run_noise_experiments

INPUT_SIZE = 5 * 7
LATENT_SIZE = 2
EPOCHS = 25000
TEST_TRIES = 100

log.basicConfig(level=log.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def compare_arrays(arr1, arr2) -> bool:
    return np.array_equal(arr1, arr2)

def save_error_to_file(arq, errors, file_path):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    errors_str = ",".join(map(str, errors))
    with open(file_path, "a") as f:
        line = f"{arq};{timestamp};{errors_str}\n"
        f.write(line)

def plot_latent_space(encoder, X, labels=None, output_path='latent_space.png'):
    """Plot latent space representations in 2D."""
    latent = encoder.get_latent_representation(X)
    
    if latent.shape[1] != 2:
        raise ValueError(f"Latent space must be 2D, got {latent.shape[1]} dimensions")
    
    plt.figure(figsize=(12, 10))
    
    if labels is not None:

        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = (labels == label)
            
            if np.sum(mask) > 0:
                plt.scatter(latent[mask, 0], latent[mask, 1], 
                           label=f'{label}' if label != '\x7f' else 'DEL', 
                           alpha=0.7, 
                           s=150,
                           c=[colors[idx]],
                           edgecolors='k',
                           linewidth=0.5)
                
        for i, label in enumerate(labels):
            display_text = 'DEL' if label == '\x7f' else label
            plt.annotate(display_text, 
                        (latent[i, 0], latent[i, 1]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=9,
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='white', 
                                 edgecolor='gray', 
                                 alpha=0.7))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  ncol=2, fontsize=8)
        
    else:
        print("No labels - plotting all points")
        plt.scatter(latent[:, 0], latent[:, 1], alpha=0.7, s=100, c='blue')

    x_range = latent[:, 0].max() - latent[:, 0].min()
    y_range = latent[:, 1].max() - latent[:, 1].min()
    x_margin = max(x_range * 0.15, 0.1)
    y_margin = max(y_range * 0.15, 0.1)
    
    plt.xlim(latent[:, 0].min() - x_margin, latent[:, 0].max() + x_margin)
    plt.ylim(latent[:, 1].min() - y_margin, latent[:, 1].max() + y_margin)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Representation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"Saving to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved successfully")

def main2():

    encoder = BasicAutoencoder(
        [35,30,25,20,15,10,5,2],

        learning_rate=0.001,
        epsilon=1e-4,
        optimizer='adam',
        activation_function='tanh',
        noise_amount=0.2,
        sigma=0.0,
        seed=42
    )

    #train
    characters = get_font3()
    binary_characters = np.array([
        to_binary_array(character).flatten() 
        for character in characters
    ])

    error_by_epoch = []

    errors = encoder.train(binary_characters, epochs=EPOCHS)

    save_error_to_file("GoodArq", errors, "outputs/errors.txt")

    plt.plot(errors)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.savefig("outputs/error_over_epochs.png")
    plt.close()

    with open("outputs/errors.txt", "w") as f:
        for epoch_errors in error_by_epoch:
            f.write(",".join(map(str, epoch_errors)) + "\n")

    errors_by_epoch = read_errors_from_file("outputs/errors.txt")
    plot_error(errors_by_epoch, output_path="outputs/error_over_epochs.png")

    #Read errors from file
    
    os.makedirs("outputs", exist_ok=True)
    # Test
    for idx, character in enumerate(binary_characters):
        result = []
        output = encoder.predict(character.reshape(1, -1))
        output_reshaped = output.reshape(7, 5)

        plot_character(output_reshaped, output_path="outputs/character_{}.png".format(idx + 1))

    labels_chars = np.array([chr(i) for i in range(0x60, 0x80)])  # Crear como numpy array

    plot_latent_space(encoder, binary_characters, labels=labels_chars, 
                    output_path="outputs/latent_space.png")
    print("saving model state...")
    encoder.save_state_pickle("autoencoder_state_{}.pkl".format(time.strftime("%H%M%S.%d.%m.%Y")))

    # --- Guardar caracteres con ruido vs reconstruidos ---
    os.makedirs("outputs/noisy_vs_predicted", exist_ok=True)

    all_originals = []
    all_noisy = []
    all_predicted = []

    for idx, character in enumerate(binary_characters):

        # Generar ruido con la misma función usada en entrenamiento,
        # pero sobre UN SOLO carácter.
        if encoder.noise_amount!=0.0:
            noisy_char = add_salt_and_pepper_noise(
                character.reshape(1, -1),
                noise_amount=encoder.noise_amount
            )
        if encoder.sigma!=0.0:
            noisy_char = add_gaussian_noise(
                character.reshape(1, -1),
                sigma=encoder.sigma
            )

        # Obtener reconstrucción del carácter ruidoso
        predicted = encoder.predict(noisy_char).reshape(7, 5)
        noisy_reshaped = noisy_char.reshape(7, 5)
        original_reshaped = character.reshape(7, 5)

        # Guardar en listas
        all_originals.append(original_reshaped)
        all_noisy.append(noisy_reshaped)
        all_predicted.append(predicted)

        fig, axes = plt.subplots(1, 3, figsize=(9, 4))

        # Reshape del character original a 7x5
        axes[0].imshow(character.reshape(7, 5), cmap='gray_r')
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(noisy_reshaped, cmap='gray_r')
        axes[1].set_title("Noisy Input")
        axes[1].axis('off')

        axes[2].imshow(predicted, cmap='gray_r')
        axes[2].set_title("Predicted Output")
        axes[2].axis('off')

        plt.tight_layout()
        fig.savefig(f"outputs/noisy_vs_predicted/comparison_{idx+1}.png", dpi=120)
        plt.close()

    rows = 6
    cols = 6
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))

    for col_idx, (images, title) in enumerate([
        (all_originals, "Original"),
        (all_noisy, "Noisy Input"),
        (all_predicted, "Predicted Output")
    ]):
        grid = np.ones((rows * 7 + (rows-1) * 2, cols * 5 + (cols-1) * 2)) * 0 
        
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
            row = i // cols
            col = i % cols
            
            y_start = row * (7 + 2)
            x_start = col * (5 + 2)
            
            grid[y_start:y_start+7, x_start:x_start+5] = img
        
        axes[col_idx].imshow(grid, cmap='gray_r')
        axes[col_idx].set_title(title, fontsize=16)
        axes[col_idx].axis('off')

    plt.tight_layout()
    fig.savefig("outputs/noisy_vs_predicted/all_comparisons_grid.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved comparison images in outputs/noisy_vs_predicted/")

if __name__ == "__main__":
    main2()
    # characters = get_font3()
    # binary_characters = np.array([
    #     to_binary_array(character).flatten() 
    #     for character in characters
    # ])

    # architectures = [
    #     [35,25,15,2],
    #     [35,30,25,20,15,10,5,2],
    #     [35,25,18,12,8,4,2],
    # ]

    # labels = np.array([chr(i) for i in range(0x60, 0x80)])

    # run_noise_experiments(binary_characters, labels, architectures)