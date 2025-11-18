

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
    with open(file_path, "a") as f:
        for epoch_error in errors:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{arq};{timestamp};{epoch_error}\n"
            f.write(line)

def main2():

    encoder = BasicAutoencoder(
        [35,25,18,12,8,4,2],
        ##[35,20,10,2],
        ##[35, 16, 8, 4, 2],
        ## probar con threadhold ( 0.5 )
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

    save_error_to_file("GoodArq", errors, "outputs/errors.txt")

    plt.plot(errors)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.savefig("outputs/error_over_epochs.png")
    plt.close()

    # with open("outputs/errors.txt", "w") as f:
    #     for epoch_errors in error_by_epoch:
    #         f.write(",".join(map(str, epoch_errors)) + "\n")

    # errors_by_epoch = read_errors_from_file("outputs/errors.txt")
    # plot_error(errors_by_epoch, output_path="outputs/error_over_epochs.png")

    # Read errors from file
    
    os.makedirs("outputs", exist_ok=True)
    # Test
    for idx, character in enumerate(binary_characters):
        result = []
        output = encoder.predict(character.reshape(1, -1))
        output_reshaped = output.reshape(7, 5)

        plot_character(output_reshaped, output_path="outputs/character_{}.png".format(idx + 1))

    print("saving model state...")
    # encoder.save_state("autoencoder_state_{}.txt".format(time.strftime("%H%M%S.%d.%m.%Y")))
    encoder.save_state_pickle("autoencoder_state_{}.pkl".format(time.strftime("%H%M%S.%d.%m.%Y")))

if __name__ == "__main__":
    main2()