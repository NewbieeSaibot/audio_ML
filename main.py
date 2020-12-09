import numpy as np
from feature.low_level_features import *
import librosa


# Constants
FILE_PATH = "./dataset/test.wav"
SECONDS_TO_CONSIDER = 1
FRAME_SIZE = 2048
HOP_LENGTH = 512
SIGNAL_RATE = 44100
NUM_SAMPLES_TO_CONSIDER = SECONDS_TO_CONSIDER * SIGNAL_RATE


# Architecture functions
def extract_temporal_features(signal, frame_size, hop_length):
    # mean, median, max (amplitude envelope), rms, zero_cross_rate

    data = {
        "mean": signal_mean(signal, frame_size, hop_length),
        "median": signal_median(signal, frame_size, hop_length),
        "amplitude_envelope": amplitude_envelope(signal, frame_size, hop_length),
        "rms": rms(signal, frame_size, hop_length),
        "zero_crossing_rate": zero_crossing_rate(signal, frame_size, hop_length),
    }

    return data


#
def extract_frequency_features(signal, sr, frame_size, hop_length):
    # band energy ratio, spectral_centroid, bandwidth, spectral_flux

    data = {
        "band_energy_ratio": band_energy_ratio(signal, frame_size, hop_length),
        "spectral_centroid": spectral_centroid(),
        "bandwidth": bandwidth(),
        "spectral_flux": spectral_flux(),
    }


def extract_time_frequency_features(signal, sr):
    pass


def extract_low_level_features(signal, sr, frame_size, hop_length):
    extract_temporal_features(signal, frame_size, hop_length)
    extract_frequency_features(signal, sr)
    extract_time_frequency_features(signal, sr)


def extract_mid_level_features(signal, sr, frame_size, hop_length):
    pitch_descriptors()
    beat_descriptors()
    note_onsets()
    fluctuation_patterns()
    mfcc()


def extract_high_level_features(signal, sr, frame_size, hop_length):
    instrumentation()
    key()
    chords_and_melody()
    rhythm()
    tempo()
    lyrycs()
    genre()
    mood()


def extract_all_features(signal, sr, frame_size, hop_length):
    extract_low_level_features(signal, sr, frame_size, hop_length)
    extract_mid_level_features(signal, sr, frame_size, hop_length)
    extract_high_level_features(signal, sr, frame_size, hop_length)


def sample_sin(sr, frequency):
    sine = np.empty(sr, dtype=float)
    for i in range(sr):
        sine[i] = np.sin(2.0*np.pi*frequency*(i/sr))
    return sine


def sample_dc_signal(sr):
    dc = np.empty(sr)
    for i in range(sr):
        dc[i] = 1
    return dc


if __name__ == '__main__':
    #signal, sr = librosa.load(FILE_PATH)
    #signal = signal[:NUM_SAMPLES_TO_CONSIDER]
    #signal, sr = librosa.load("./dataset/KeyC.wav")
    signal = sample_sin(SIGNAL_RATE, 1500) + sample_sin(SIGNAL_RATE, 2100)

    #signal2 = sample_dc_signal(SIGNAL_RATE)

    spectrum = np.abs(np.fft.fft(signal))
    #print(len(spectrum))
    for i in range(len(spectrum)//2):
        print("freq: ", i*(SIGNAL_RATE/len(spectrum)), " mag: ", spectrum[i])

    #ber = band_energy_ratio(signal, SIGNAL_RATE, FRAME_SIZE, HOP_LENGTH, 2000)
    #for i in range(len(ber)):
    #    print(ber[i])
    #extract_all_features(signal, SIGNAL_RATE, FRAME_SIZE, HOP_LENGTH)

