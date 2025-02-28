import numpy as np
import matplotlib.pyplot as plt


def my_gaussian_filter(size, sigma):
    n = np.arange(size)
    mu = size / 2
    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((n - mu) ** 2) / (2 * sigma ** 2))

    return g


def gabor_filter(size, sigma, freq):
    n = np.arange(size)
    cos_h = my_gaussian_filter(size, sigma) * np.cos(2 * np.pi * freq * n)
    sin_h = my_gaussian_filter(size, sigma) * np.sin(2 * np.pi * freq * n)

    return cos_h, sin_h


def plot_gabor_filter(gabor_cos, gabor_sin, size):
    t = np.arange(size)

    plt.figure()
    plt.grid()
    plt.xlim(0, 1200)
    plt.title('Gabor filter - cos')
    plt.ylabel('x10^(-3)', loc='top')
    plt.plot(t, gabor_cos * (10 ** 3))
    plt.savefig('Budulan_Andreea_Cristina_343C2_gabor_cos.png')
    plt.show()
    plt.title('Gabor filter - sin')
    plt.ylabel('x10^(-3)', loc='top')
    plt.grid()
    plt.xlim(0, 1200)
    plt.plot(t, gabor_sin * (10 ** 3))
    plt.savefig('Budulan_Andreea_Cristina_343C2_gabor_sin.png')
    plt.show()