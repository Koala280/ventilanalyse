from scipy.io import wavfile
import noisereduce as nr
import os

def denoise_audio(file_path):
    file_name = os.path.split(file_path)[1]
    # load data
    rate, data = wavfile.read(file_path)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    

    EXPORT_PATH = "audios/filtered_audio/denoised"

    if not os.path.exists(EXPORT_PATH):
        os.mkdir(EXPORT_PATH)

    wavfile.write(f"{EXPORT_PATH}/denoised_{file_name}", rate, reduced_noise)