

import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from autoenconder import BasicAutoencoder
import logging as log
from read_utils import plot_character, get_font3, to_binary_array

INPUT_SIZE = 5 * 7
LATENT_SIZE = 2
EPOCHS = 1000
TEST_TRIES = 100

log.basicConfig(level=log.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def compare_arrays(arr1, arr2) -> bool:
    return np.array_equal(arr1, arr2)

def main():

    encoder = BasicAutoencoder(
        [35, 16, 8, 2],
        learning_rate=0.01,
        epsilon=1e-4,
        optimizer='sgd',
        activation_function='sigmoid',
        seed=42
    )

    #train
    characters = get_font3()
    # binary_characters = [to_binary_array(character) for character in characters]
    binary_characters_flattened = [character.reshape(1, -1) for character in (to_binary_array(character) for character in characters)]


#     for character in characters:
#         binary = to_binary_array(character)
#         flattened = binary.reshape(1, -1)
#         print(
# f"""character:
# {character}
# Binary:
# {binary}
# Flattened:
# {flattened}
# """
#                 )
        
    print("Initializing from file...")
    encoder.initialize_from_file("outputs/autoencoder_state_1763338301.969575.txt")
    

    for epoch in tqdm.tqdm(range(100000), desc="Training Autoencoder"):
        for character_flattened in binary_characters_flattened:
            #print("Training on character: {}".format(character_flattened))
            encoder.train(character_flattened, character_flattened, 1)



    #test 
    for idx, character in enumerate(binary_characters_flattened):
        result = []
        output = encoder.predict(character)
        # for value in output.flatten():
        #     result.append(1 if value >= 0.5 else 0)
        # result_array = np.array(result).reshape(5, 7)
        # print("Original Character {}:".format(idx + 1))
        output = output.reshape(7, 5)
        # print(character)
        plot_character(output, output_path="outputs/character_{}.png".format(idx + 1))

    encoder.save_state("autoencoder_state_{}.txt".format(time.time()))



if __name__ == "__main__":
    main()