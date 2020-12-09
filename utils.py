import numpy as np

# Utils


def sgn(number):
    if number >= 0:
        return 1
    else:
        return -1


# Hann function, or triangle, probably I will make a class with windowing functions
def windowing_function(signal, type="Hann"):
    # Hann implementation
    window = np.empty(len(signal))
    for i in range(len(signal)):
        window[i] = signal[i]
    if type == "Hann":
        for i in range(len(window)):
            window[i] = window[i] * 0.5 * (1 - np.cos((2*np.pi*i)/(len(window))))
    elif type == "Nutall":
        for i in range(len(window)):
            window[i] = window[i] * (0.355768 - 0.487396*np.cos(2*np.pi*i/len(window)) + 0.144232*np.cos(4*np.pi*i/len(window)) - 0.012604*np.cos(6*np.pi*i/len(window)))
    else:
        return window

    return window


def hz_to_mel(frequency):
    return 1127*np.log10(1 + frequency/700)


def hz_to_bark(frequency):
    return ((26.81*frequency)/(1960 + frequency)) - 0.53
