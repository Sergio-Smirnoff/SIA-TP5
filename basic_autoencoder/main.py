from seaborn import heatmap
import numpy as np
import matplotlib.pyplot as plt


from read_utils import plot_character, get_font3, to_binary_array

INPUT_SIZE = 5 * 7
LATENT_SIZE = 2
EPOCHS = 1000
TEST_TRIES = 100

def main():



    # encoder = BasicAutoencoder()

    #train
    characters = get_font3()
    binary_characters = [to_binary_array(character) for character in characters]

    for epoch in range(EPOCHS):
        for binary_character in binary_characters:
            #encoder.train()
            pass

    #test 
    results = [0]*32 #one for each pattern

    for i in range(TEST_TRIES):
        for idx, character in enumerate(binary_characters):
            # result = encoder.forward()
            results[idx] += 1 if result == character else 0
            
            #plot some results
            if i % (TEST_TRIES/10) == 0:
                plot_character(character=result, output_path="output_char_{}_try_{}.png".format(idx, i))
                

    results = [res / TEST_TRIES for res in results]
    print("Reconstruction accuracy over {} tries:".format(TEST_TRIES))
    for idx, acc in enumerate(results):
        print("Character {}: {:.2f}%".format(idx, acc * 100))

    # plot_character(character=5, output_path="output.png")
    

if __name__ == "__main__":
    main()