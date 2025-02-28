import numpy as np

# F grupuri de K esantioane
def create_windows(audio, k, delta_samples):
    # dimensiune(audio)
    audio_1d = audio.ravel()
    N = len(audio_1d)
    F = (N + delta_samples) // (k + delta_samples)

    # creez F fereste de cate K esantioane la distanta de 12ms unele de altele
    windows = []
    start = 0
    end = k

    for i in range(F):
        window = audio_1d[start:end]
        windows.append(window)

        start = end + delta_samples
        end = start + k

    return np.array(windows)