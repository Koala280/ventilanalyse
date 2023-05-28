import os

AUDIO_DIR = "audios/labeled/"
dict_genres = {'positive': 0, 'negative': 1, 'noise': 2}

def get_audio_path(track_id):
    for label in dict_genres.keys():
        for folder in os.listdir(AUDIO_DIR + label):
            for file in os.listdir(AUDIO_DIR + label + "/" + folder):
                if file[:-4] == track_id:
                    return AUDIO_DIR + label + "/" + folder + "/" + file