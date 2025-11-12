

import numpy as np
import matplotlib.pyplot as plt
from autoenconder import BasicAutoencoder
import logging as log
from read_utils import plot_character, get_font3, to_binary_array

INPUT_SIZE = 5 * 7
LATENT_SIZE = 2
EPOCHS = 1000
TEST_TRIES = 100

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    binary_characters = [to_binary_array(character) for character in characters]


    binary_characters_flattened = [character.reshape(1, -1) for character in binary_characters]
    for epoch in range(1):
        for character_flattened in binary_characters_flattened:
            #print("Training on character: {}".format(character_flattened))
            encoder.train(character_flattened, character_flattened)

    #test 
    results = [0]*32 #one for each pattern

    #[ 0, 0, 0, 0 .... 0]

    # for i in range(1):
    for idx, character in enumerate(binary_characters):

        result = []
        output = encoder.predict(character.reshape(1, -1))

        for i in range(len(output)):
            for j in range(len(output[i])):
                result.append(1 if output[i][j] >= 0.5 else 0)
                
        print("Result for character {}: {}".format(idx, result))
        
        #characters estan reshapeados a 1d array
        #result se outputtea como 1d array
        # results[idx] += 1 if  int(result[0]) == idx else 0

        #plot some results
        # if i % (TEST_TRIES/10) == 0:
            # plot_character(character=characters[result], output_path="outputs/output_char_{}_try_{}.png".format(idx, i))
        results.append(result)

    for idx, coso in enumerate(results):
        print(coso)
        plot_character(character=idx, output_path="outputs/output_char_{}_try_{}.png".format(idx, i))


if __name__ == "__main__":
    main()