import os
import matplotlib.pyplot as plt
import numpy as np
import wave

# Name der wav-Datei
recName = "rec8"

# Lade die WAV-Datei
wav_file = wave.open(os.path.join(os.path.dirname(__file__), f"..\\..\\recordings\\{recName}.wav"), "r")
signal = wav_file.readframes(-1)
signal = np.frombuffer(signal, dtype='int16')

# Berechne die FFT des Signals
fft = np.fft.fft(signal)

# Erstelle eine Frequenzachse
sample_rate = wav_file.getframerate()
fft_size = len(signal)
freqs = np.fft.fftfreq(fft_size, 1/sample_rate)


# TESTAUSGABEN
#--------------
# print(sample_rate)
# print(fft_size)
# test = wav_file.readframes(-1)
# print(signal)
#--------------



# Schneide das Ergebnis in die Hälfte, da die FFT spiegelbildlich ist
half_freqs = freqs[:int(len(freqs)/2)]
half_fft = np.abs(fft[:int(len(fft)/2)])

# Plotte das Ergebnis
plt.plot(half_freqs, half_fft)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# # Speichere fft Plot
# # plt.savefig(os.path.join(os.path.dirname(__file__), "..\\vis\\" + recName + '.png'))
