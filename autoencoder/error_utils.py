import numpy as np
import matplotlib.pyplot as plt

def compare_pixel_error(input_character: np.array, predicted_character: np.array) -> int:
    """ Returns the number of differing pixels between two characters represented as binary arrays.
        Args:
            input_character (np.array): Original character as a flatten binary array
            predicted_character (np.array): Reconstructed character as a flatten binary array.
        Returns:
            int: Number of differing pixels.
    """

    input_flat = input_character.flatten()
    predicted_flat = predicted_character.flatten()

    result = 0
    for idx, pixel in enumerate(input_flat):
        if pixel != predicted_flat[idx]:
            result += 1
    return result


def read_errors_from_file(file_path: str) -> list:
    """ Reads error values from a file.
        Args:
            file_path (str): Path to the file containing error values.
        Returns:
            list: List of error values per epoch.
    """
    errors_by_epoch = []
    with open(file_path, "r") as f:
        for line in f:
            epoch_errors = list(map(int, line.strip().split(",")))
            errors_by_epoch.append(epoch_errors)
    return errors_by_epoch

def plot_error(errors_by_epoch: list, output_path: str):
    """ Plots the error over epochs and saves it to a file.
        Args:
            errors_by_epoch (list): List of error values per epoch.
            output_path (str): Path to save the plotted error graph.
    """
    for character_errors in errors_by_epoch:
        plt.plot(character_errors, label='Pixel Error')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Differing Pixels')
    plt.title('Pixel Error Over Epochs')
    # plt.legend()
    plt.grid()
    plt.savefig(output_path)
