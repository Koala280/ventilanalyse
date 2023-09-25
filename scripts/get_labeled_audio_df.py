import os
import pandas as pd

AUDIO_DIR = "audios/labeled/"
classes = {'positive': 0, 'negative': 1}

def get_labeled_audio_df():
    data = []

    for label in classes.keys():
        for folder in os.listdir(AUDIO_DIR + label):
            for file in os.listdir(AUDIO_DIR + label + "/" + folder):
                file_path = AUDIO_DIR + label + "/" + folder + "/" + file
                positive = int(label == "positive")
                negative = int(label == "negative")
                data.append((file_path, positive, negative))

    file_path, positive, negative = zip(*data)
    df = pd.DataFrame({"file_path": file_path, "positive": positive, "negative": negative})
    
    return df