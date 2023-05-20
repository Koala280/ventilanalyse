from scipy.io import wavfile
import noisereduce as nr
import os

def denoise(file_path, prop_decrease=1):
    file_name = os.path.split(file_path)[1]
    EXPORT_PATH = "audios/filtered_audio/denoised"
    EXPORT_FILE = f"{EXPORT_PATH}/denoised_{file_name}"

    if not os.path.exists(EXPORT_PATH):
        os.mkdir(EXPORT_PATH)

    # load data
    rate, data = wavfile.read(file_path)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=prop_decrease)

    wavfile.write(EXPORT_FILE, rate, reduced_noise)

    return EXPORT_FILE
