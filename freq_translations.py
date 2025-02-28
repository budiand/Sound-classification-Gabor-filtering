import numpy as np


def mel_translation(normal_freq):
    mel_freq = 1127 * np.log(1 + (normal_freq/700))

    return mel_freq

def normal_translation(mel_freq):
    normal_freq = 700*(np.exp(mel_freq/1127) - 1)

    return normal_freq