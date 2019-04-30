
import matplotlib.pyplot as plt
from sys import argv

'''

Created: 29 April 2019, Tyler Amos

Usage:
        python plot_results.py log_likelihoodfile.csv out_plot.png
'''


if __name__=="__main__":

    log_likes = []
    with open(argv[1]) as f:
        for ll in f:
            log_likes.append(float(ll))

    plt.plot(range(len(log_likes)), log_likes)
    plt.title("Log likelihood as a function of iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")
    plt.savefig(argv[2])
