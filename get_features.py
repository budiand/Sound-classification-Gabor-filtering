import numpy as np
from matplotlib.pyplot import plot_date
from sklearn.neighbors import KNeighborsClassifier
import scipy
import matplotlib.pyplot as plt
from gabor_filter import *
from freq_translations import *
from plot_spectrum import *
from create_windows import *
from filter_windows import filter_windows


def get_features(audio_train, fs):
    """
    Functie care calculeaza trasaturile unui semnal audio.
    :param audio: matrice de dimensiune [D x N], unde D este numarul de fisiere audio si N este numarul de esantioane
    :param fs: frecventa de esantionare a semnalului audio
    :return: matrice de dimensiune [D x (2*M)], unde M este numarul de trasaturi calculate
    """
    size = 1102
    sigma = 187.21221
    freq = 0.00267

    # EX 2

    M = 12  # nr de segmente
    A = mel_translation(0)  # frecv corespunzatoare 0 Hz pe scala normala
    B = mel_translation(fs / 2)  # frecv corespunzatoare fs/2 Hz pe scala normala

    mel_freq = np.linspace(A, B, M + 1)  # 12 segmente egal departate(MEL)
    normal_freq = [normal_translation(freq) for freq in mel_freq]
    l = [normal_freq[i] - normal_freq[i - 1] for i in range(1, M + 1)]
    c = [(normal_freq[i] + normal_freq[i + 1]) / 2 for i in range(0, M)]

    gabor_filters = [gabor_filter(size, fs / l[i], c[i] / fs) for i in range(0, M)]

    plot_gabor_filter(gabor_filters[0][0], gabor_filters[0][1], size)

    # EX 3

    plot_spectrum(gabor_filters, M, size)

    # EX 4
    k = size
    delta = 12 * 10 ** -3
    delta_samples = int(fs * delta)

    # despart filtrele in cele doua componente separate, cos si sin
    # pentru a obtine matricea de forma [M, K]
    gabor_filters_cos = []
    gabor_filters_sin = []
    for i in range(M):
        cos_h, sin_h = gabor_filter(size, fs / l[i], c[i] / fs)

        gabor_filters_cos.append(cos_h)
        gabor_filters_sin.append(sin_h)

    gabor_matrix_cos = np.array(gabor_filters_cos)
    gabor_matrix_sin = np.array(gabor_filters_sin)

    feat_train = []
    for audio in audio_train:
        # matrice de dimensiune [F, K]
        windows = create_windows(audio, k, delta_samples)

        # se transpune rezultatul pentru a obtine o matrice de forma [F, M]
        filtered_windows_cos = filter_windows(windows, gabor_matrix_cos).T
        filtered_windows_sin = filter_windows(windows, gabor_matrix_sin).T

        filtered_windows_cos = np.abs(filtered_windows_cos)
        filtered_windows_sin = np.abs(filtered_windows_sin)

        mean_features_combined = (np.mean(filtered_windows_cos, axis=0) + np.mean(filtered_windows_sin, axis=0)) / 2
        std_features_combined = (np.std(filtered_windows_cos, axis=0) + np.std(filtered_windows_sin, axis=0)) / 2

        # [2M]
        feature_vector = np.concatenate([mean_features_combined, std_features_combined])
        feat_train.append(feature_vector)

    # [D, 2M]
    return np.array(feat_train)