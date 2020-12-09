import librosa
import numpy as np
import utils

NOTES = [261, 297, 330, 352, 396, 440, 495,
         528, 594, 660, 704, 792, 880, 990,
         1056]

frequency_to_note = {
    261: "C4", 297: "D4", 330: "E4", 352: "F4", 396: "G4", 440: "A4", 495: "B4",
    528: "C5", 594: "D5", 660: "E5", 704: "F5", 792: "G5", 880: "A5", 990: "B5",
    1056: "C6",
}
notes_through_time = []
sr = 44100


def score_generator(filepath, bpm, minimum_interval):
    signal = librosa.load(filepath)[0]
    window_s = (1 / (bpm / 60)) * minimum_interval * sr
    window_size = int(window_s)
    hop_length = int(window_s)
    print("window size: ", window_s, "frequency interval: ", (sr/2)/window_s)

    for i in range(0, len(signal), hop_length):
        print("window " + str(i))

        window = utils.windowing_function(signal[i:i + window_size], "Hann")
        spectrum = np.abs(np.fft.fft(window))

        '''
        print("total energy:", np.sum(spectrum))
        
        for j in range(len(spectrum)//2):
            print("freq: ", j*(sr/len(spectrum)), " mag: ", spectrum[j])
        '''

        window_features = feature_extractor(spectrum)
        window_notes = decision_taker(window_features)
        notes_through_time.append((i, window_notes))

    print(notes_through_time)


def feature_extractor(spectrum):
    features = {'total_energy': np.sum(spectrum), "energy": [], "coherence": []}

    for i in range(len(NOTES)):
        harmonica = 1
        energy = 0
        coherence = 1
        while (harmonica*NOTES[i]) <= sr//2:
            if harmonica > 4:
                break
            for frequency_deviation in range(0, 1, 1):
                energy += spectrum[(((NOTES[i]+frequency_deviation)*harmonica)*len(spectrum))//sr]*(1/harmonica)
                coherence *= spectrum[(((NOTES[i]+frequency_deviation)*harmonica)*len(spectrum))//sr]*(1/harmonica)
            harmonica += 1

        features['coherence'].append(coherence)
        features['energy'].append(energy)

    return features


def decision_taker(window_features):
    frame_notes = []

    # naive classifier with dynamic thresholds
    counted_energy = np.sum(np.array(window_features['energy']))
    energy_threshold = counted_energy/5
    counted_threshold = np.sum(np.array(window_features['coherence']))
    coherence_threshold = counted_threshold/9

    print("energy_threshold", energy_threshold, "coherence_threshold", coherence_threshold)
    for i in range(len(window_features['energy'])):
        print(frequency_to_note[NOTES[i]], "energy:", window_features['energy'][i], "coherence:",
              window_features['coherence'][i])
        if window_features['energy'][i] >= energy_threshold and window_features['coherence'][i] >= coherence_threshold:
            frame_notes.append(frequency_to_note[NOTES[i]])

    return frame_notes[:]


def note_detector():
    pass


score_generator('dataset/KeyCsaw.wav', 130, 1/4)
