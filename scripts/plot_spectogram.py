from scripts.create_spectogram import create_spectogram
import matplotlib.pyplot as plt
import librosa

def plot_spectogram(file_path):
    spect = create_spectogram(file_path)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()