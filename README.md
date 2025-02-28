# Sound Classification and Detection - Using the Gabor Filter

### Obtained Accuracy:

- Train: 0.75
- Test: 0.65

## 1. gabor_filter(size, sigma, freq)

- The **gabor_filter** function implements a 1D Gabor filter using a Gaussian filter and sinusoids, with a specified size and frequency.

## 2. Translation from Normal Scale to Mel Scale and Creating a Set of Filters

- In the file **freq_translations.py**, the functions that perform the translation from a normal scale to a Mel scale and vice versa are implemented. These functions ensure a set of equally spaced frequencies on the Mel scale, which are perceived as equally spaced audibly by humans.
- Using the resulting frequencies, a set of M = 12 Gabor filters centered at these frequencies is created.

## 3. Spectrum of the Filters

- The **plot_spectrum(gabor_filters, M, size)** function calculates the positive spectrum (using FFT) for the cos_h and sin_h components of the filters in the **gabor_filters** set and displays both the cos_h and sin_h components using the plt.plot() function.

## 4.a create_windows(audio, k, delta_samples)

- This function calculates the number of F windows from the total number of N ~ 10^6 samples in an audio file. We assumed that the F windows cover the entire range of samples, leading to the following formula for calculating F:

> F = (N + deltaSamples) // (K + deltaSamples), where:  
> K = size(window), deltaSamples = number of samples between windows.

## 4.b filter_windows(filters, windows)

- The **filter_windows(filters, windows)** function is responsible for applying the filters to the windows of samples extracted from the audio signal.

## get_features(audio_train, fs)

- In the **get_features(audio_train, fs)** function, all the previously discussed functions are applied to obtain the feature matrix corresponding to the audio dataset.
