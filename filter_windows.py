import numpy as np

def filter_windows(filters, windows):
    filters_inverse = filters[:, ::-1]  # Inversarea coloanelor
    filters_transpose = np.transpose(filters_inverse)  # Transpunerea
    filtered_windows = np.dot(windows, filters_transpose)

    return filtered_windows
