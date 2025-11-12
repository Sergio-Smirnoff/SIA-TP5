from seaborn import heatmap
import numpy as np
import matplotlib.pyplot as plt

from read_utils import read_character

def main():
    heatmap(
        read_character(2), 
        linewidths=0.2, 
        cbar=False, 
        square=True,
        cmap=plt.get_cmap('binary'), 
        linecolor='k')
    plt.show()

if __name__ == "__main__":
    main()