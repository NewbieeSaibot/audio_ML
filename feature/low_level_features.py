import numpy as np
from utils import windowing_function, sgn

# Low Level Features

# Time Domain Features


def signal_mean(signal, frame_size, hop_length):
    means = []
    for i in range(0, len(signal), hop_length):
        means.append(signal[i:i + frame_size].mean())

    return means


def signal_median(signal, frame_size, hop_length):
    medians = []
    for i in range(0, len(signal), hop_length):
        medians.append(signal[i:i + frame_size].medians())

    return medians


def amplitude_envelope(signal, frame_size, hop_length):
    envelope = []
    for i in range(0, len(signal), hop_length):
        envelope.append(signal[i:i + frame_size].max())

    return envelope


def rms(signal, frame_size, hop_length):
    rmss = []
    for i in range(0, len(signal), hop_length):
        rms = 0
        for j in range(frame_size):
            rms += signal[i + j] ** 2

        rmss.append((rms / frame_size) ** 0.5)

    return rmss


def zero_crossing_rate(signal, frame_size, hop_length):
    crosses = []
    for i in range(0, len(signal), hop_length):
        aux = 0
        for j in range(frame_size - 1):
            aux += abs(sgn(signal[i + j]) - sgn(signal[i + j] + 1))
        crosses.append(aux / 2)

    return crosses


# Frequency Domain Features

def band_energy_ratio(signal, sr, frame_size, hop_length, cutoff_frequency):
    band_energies = []

    for i in range(0, len(signal), hop_length):
        windowed_signal = windowing_function(signal[i:i+frame_size])
        spectrum = np.abs(np.fft.fft(windowed_signal))
        band1 = 0
        band2 = 0
        for j in range(len(spectrum)//2):
            if j*(sr/len(spectrum)) < cutoff_frequency:
                band1 += spectrum[j]**2
            else:
                band2 += spectrum[j]**2

        band_energies.append(band1/band2)

    return band_energies


def spectral_centroid(signal, frame_size, hop_length):
    spectral_centroids = []

    for i in range(0, len(signal), hop_length):
        windowed_signal = windowing_function(signal[i:i+frame_size])
        spectrum = np.abs(np.fft.fft(windowed_signal))
        numerator = 0
        denominator = 0
        for j in range(len(spectrum)//2):
            numerator += spectrum[j]*j
            denominator += spectrum[j]
        spectral_centroids.append(numerator/denominator)

    return spectral_centroids


def bandwidth(signal, frame_size, hop_length):
    bandwidths = []
    spectral_centroids = []
    # if you already has calculated the spectral centroid is just waste of time.
    for i in range(0, len(signal), hop_length):
        windowed_signal = windowing_function(signal[i:i+frame_size])
        spectrum = np.abs(np.fft.fft(windowed_signal))
        numerator = 0
        denominator = 0
        for j in range(len(spectrum)//2):
            numerator += spectrum[j]*j
            denominator += spectrum[j]
        spectral_centroids.append(numerator/denominator)

    for i in range(0, len(signal), hop_length):
        windowed_signal = windowing_function(signal[i:i+frame_size])
        spectrum = np.abs(np.fft.fft(windowed_signal))
        numerator = 0
        denominator = 0
        for j in range(len(spectrum)//2):
            numerator += np.abs(j - spectral_centroids[j])*spectrum[j]
            denominator += spectrum[j]
        bandwidths.append(numerator/denominator)

    return bandwidths


def spectral_flux(signal, frame_size, hop_length):
    spectral_fluxes = []

    for i in range(0, len(signal), hop_length):
        windowed_signal = windowing_function(signal[i:i+frame_size])
        spectrum = np.abs(np.fft.fft(windowed_signal))
        spectrum /= sum(spectrum)
        aux = 0
        for j in range(len(spectrum)//2):
            aux += (spectrum[j+1]-spectrum[j])**2
        spectral_fluxes.append(aux)

    return spectral_fluxes
