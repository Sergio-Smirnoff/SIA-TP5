from read_utils import plot_character, get_font3, to_binary_array
from vae import VAE
import numpy as np


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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Preparing dataset...")
    X = prepare_dataset()

    print("Initializing VAE...")
    vae = VAE(
        architecture=[35, 16, 8, 4, 2],   # input 35 → latent 2
        learning_rate=0.005,
        optimizer="adam",
        activation_function="relu",
        seed=42
    )

    print("Training...")
    epochs = 50000
    vae.train(X, epochs)
    print("\nTraining finished!")

    # ---------------------------------------------------------
    # Show reconstruction of a sample char
    # ---------------------------------------------------------
    idx = 5
    print("\nOriginal:")
    show_char(X[idx])

    print("\nReconstruction:")
    acts_enc, zv_enc, acts_dec, zv_dec = vae.forward(X[[idx]])
    show_char(acts_dec[-1][0])