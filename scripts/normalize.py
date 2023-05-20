import librosa
import librosa.display
import soundfile as sf
import os

def normalize(file_path):
    file_name = os.path.split(file_path)[1]
    EXPORT_PATH = "audios/filtered_audio/normalized"
    EXPORT_FILE = f'{EXPORT_PATH}/normalized_{file_name}'

    if not os.path.exists(EXPORT_PATH):
        os.mkdir(EXPORT_PATH)

    # Laden der Audio-Datei
    audio, sample_rate = librosa.load(file_path)

    # Normalisieren der Audio-Datei
    normalized_audio = librosa.util.normalize(audio)

    # Speichern der normalisierten Audio-Datei
    sf.write(EXPORT_FILE, normalized_audio, sample_rate)
    
    return EXPORT_FILE
