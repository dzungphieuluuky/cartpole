import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    results = np.load('results.npy')
    iterations = results[0]
    losses = results[1]
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training progress')
    plt.legend()
    plt.grid()

    plt.savefig('results.png')
    plt.show()

if __name__ == "__main__":
    plot_results()
