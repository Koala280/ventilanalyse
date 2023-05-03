import wave
import numpy as np
import matplotlib.pyplot as plt

def fft(file_name):
    with wave.open(file_name, 'r') as wav_file:
        # Extrahieren von Informationen aus der Wave-Datei
        frames = wav_file.readframes(-1)
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

    # Umwandeln von Byte-Daten in NumPy-Array
    frames = np.frombuffer(frames, dtype=np.int16)

    # Normalisieren der Daten auf den Bereich [-1, 1]
    frames = frames / 2**(8*sample_width-1)

    # Berechnen der Fourier-Transformation
    freq = np.fft.rfftfreq(len(frames), d=1/sample_rate)
    freq_amp = np.abs(np.fft.rfft(frames))
    print(f"{file_name} - sample_rate: {sample_rate} - num_channels: {num_channels} - sample_width: {sample_width}")

    # Plotten der Frequenzamplitude
    plt.figure()
    plt.plot(freq, freq_amp)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f"Frequency Amplitude")
    # plt.savefig(f"fourier\\{file_name}\\fourier.png") # TODO create folder if not existing
    plt.show()