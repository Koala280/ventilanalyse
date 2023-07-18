import os
import pandas as pd

classes = {'positive': 0, 'negative': 1}

def get_dataframe(audio_dir="audios/labeled/"):
    data = []

    for label in classes.keys():
        for folder in os.listdir(audio_dir + label):
            for file in os.listdir(audio_dir + label + "/" + folder):
                file_path = audio_dir + label + "/" + folder + "/" + file
                positive = int(label == "positive")
                negative = int(label == "negative")
                data.append((file_path, positive, negative))

    file_path, positive, negative = zip(*data)
    df = pd.DataFrame({"file_path": file_path, "positive": positive, "negative": negative})
    
    return df

def get_balanced_dataframe():
    df = get_dataframe()
    positive_rows = df[df['positive'] == 1]
    negative_rows = df[df['negative'] == 1]

    # Get the count of positive and negative rows
    positive_count = len(positive_rows)
    negative_count = len(negative_rows)

    # Randomly sample negative rows to match the count of positive rows
    balanced_negative_rows = negative_rows.sample(n=positive_count, replace=True)

    # Concatenate positive rows and balanced negative rows
    balanced_df = pd.concat([positive_rows, balanced_negative_rows])

    # Shuffle the DataFrame
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    return balanced_df