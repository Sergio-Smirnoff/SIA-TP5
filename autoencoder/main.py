

import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from error_utils import compare_pixel_error, plot_error, read_errors_from_file
from autoenconder import BasicAutoencoder
import logging as log
from read_utils import plot_character, get_font3, to_binary_array

INPUT_SIZE = 5 * 7
LATENT_SIZE = 2
EPOCHS = 100000
TEST_TRIES = 100

log.basicConfig(level=log.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def compare_arrays(arr1, arr2) -> bool:
    return np.array_equal(arr1, arr2)

# def main():

#     encoder = BasicAutoencoder(
#         [35, 16, 8, 4, 2],
#         learning_rate=0.1,
#         epsilon=1e-4,
#         optimizer='sgd',
#         activation_function='tanh',
#         seed=42
#     )

#     #train
#     characters = get_font3()
#     characters = characters[:1]  # use only first 10 characters
#     # binary_characters = [to_binary_array(character) for character in characters]
#     binary_characters_flattened = [character.reshape(1, -1) for character in (to_binary_array(character) for character in characters)]

        
#     # print("Initializing from file...")
#     # encoder.initialize_from_file_pickle("outputs/autoencoder_state_103939.17.11.2025.pkl")

#     error_by_epoch = []

#     for epoch in tqdm.tqdm(range(EPOCHS), desc="Training Autoencoder"):

#         error_by_character = []

#         for character_flattened in binary_characters_flattened:
#             encoder.train(character_flattened, character_flattened, 1)

#             #===== compute error for graphs =====
#             output = encoder.predict(character_flattened)
#             error = compare_pixel_error(character_flattened, output)
#             error_by_character.append(error)
#             #====================================
#         error_by_epoch.append(error_by_character)

#     with open("outputs/errors.txt", "w") as f:
#         for epoch_errors in error_by_epoch:
#             f.write(",".join(map(str, epoch_errors)) + "\n")

#     # errors_by_epoch = read_errors_from_file("outputs/errors.txt")
#     # plot_error(errors_by_epoch, output_path="outputs/error_over_epochs.png")

#     # Read errors from file
    

#     # Test
#     for idx, character in enumerate(binary_characters_flattened):
#         result = []
#         output = encoder.predict(character)
#         output_reshaped = output.reshape(7, 5)

#         plot_character(output_reshaped, output_path="outputs/character_{}.png".format(idx + 1))

#     print("saving model state...")
#     # encoder.save_state("autoencoder_state_{}.txt".format(time.strftime("%H%M%S.%d.%m.%Y")))
#     encoder.save_state_pickle("autoencoder_state_{}.pkl".format(time.strftime("%H%M%S.%d.%m.%Y")))

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

    ## Hiperparametros claves: [35,25,18,12,8,4,3], learning_rate=0.001, epsilon=1e-4, optimizer='adam', activation_function='tanh', seed=42

    encoder = BasicAutoencoder(
        ##[35,25,18,12,8,4,2],
        [35,20,10,2],
        ##[35, 16, 8, 4, 2],
        ## probar con threshold ( 0.5 )
        ## Probar con 3 / 4 en la capa intermedia
        learning_rate=0.001,
        epsilon=1e-4,
        optimizer='adam',
        activation_function='tanh',
        seed=42
    )

    #train
    characters = get_font3()
    #characters = [characters[-1]]  
    # binary_characters = [to_binary_array(character) for character in characters]
    binary_characters = np.array([
        to_binary_array(character).flatten() 
        for character in characters
    ])


        
    # print("Initializing from file...")
    # encoder.initialize_from_file_pickle("outputs/autoencoder_state_103939.17.11.2025.pkl")

    error_by_epoch = []

    errors = encoder.train(binary_characters, epochs=EPOCHS)

    save_error_to_file("adam", errors, "outputs/errors.txt")

    plt.plot(errors)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.savefig("outputs/error_over_epochs.png")
    plt.close()

    # with open("outputs/errors.txt", "a") as f:
    #     for epoch_errors in error_by_epoch:
    #         f.write(",".join(map(str, epoch_errors)) + "\n")

    # errors_by_epoch = read_errors_from_file("outputs/errors.txt")
    # plot_error(errors_by_epoch, output_path="outputs/error_over_epochs.png")

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
    # encoder.save_state("autoencoder_state_{}.txt".format(time.strftime("%H%M%S.%d.%m.%Y")))
    encoder.save_state_pickle("autoencoder_state_{}.pkl".format(time.strftime("%H%M%S.%d.%m.%Y")))

if __name__ == "__main__":
    main2()