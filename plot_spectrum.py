import numpy as np
import matplotlib.pyplot as plt
import scipy


def plot_spectrum(gabor_filters, M, size):
    fx = np.linspace(0, size//2, size//2)

    plt.title("Gabor filters")
    plt.grid()

    for i in range(0, M):
        coefs_cos = scipy.fft.fft(gabor_filters[i][0])[:size//2]
        coefs_sin = scipy.fft.fft(gabor_filters[i][1])[:size//2]
        plt.plot(fx, np.abs(coefs_cos))
        plt.plot(fx, np.abs(coefs_sin))

    plt.xlim(0, 600)
    plt.ylim(0, 0.5)
    plt.savefig('Budulan_Andreea_Cristina_343C2_spectru_filtre.png')
    plt.show()