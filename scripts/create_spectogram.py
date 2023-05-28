import librosa

def create_spectogram(audio_path):
    y, sr = librosa.load(audio_path)
    spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect_db = librosa.power_to_db(spect, ref=np.max)
    return spect_db.T