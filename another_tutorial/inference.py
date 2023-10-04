import os
import argparse
import tensorflow as tf
import tensorflow_io as tfio
from glob import glob
import plotly.express as px
import pandas as pd

# Model and test WAV file paths
MODEL = "audio_classification_model_old.h5"
# Load the saved model
model = tf.keras.models.load_model("./another_tutorial/"+MODEL)
TEST_WAV_FOLDER = r"C:\Users\tomsb\OneDrive\Dokumente\study\ventilanalyse\another_tutorial\audios\inference"  # Folder containing test WAV files

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess_prediction(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([16000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def preprocess_audio(audio_path):
    # Load and preprocess the audio file
    wav = load_wav_16k_mono(audio_path)

    # Pad the audio with zeros to reach the target length if it's shorter
    if len(wav) < 16001:
        zero_padding = tf.zeros([16001] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav],0)

    wav = tf.keras.utils.timeseries_dataset_from_array(
        wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1
    )

    wav = wav.map(preprocess_prediction)
    wav = wav.batch(64)
    return wav

# Function to make predictions on audio
def predict_audio(audio_slices, threshold):
    yhat = model.predict(audio_slices)
    predictions = [1 if prediction > threshold else 0 for prediction in yhat]
    return yhat, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification Inference")
    parser.add_argument("--wav", type=str, default=TEST_WAV_FOLDER, help="Path to the audio file to classify")

    parser.add_argument(
        "--val",
        type=float,
        default=0.5,
        help="Threshold for classification (default: 0.5)",
    )
    args = parser.parse_args()

    # Get a list of all WAV files in the specified folder
    test_wav_files = glob(os.path.join(TEST_WAV_FOLDER, "*.wav"))
    yhat_values = []  # List to store yhat values
    labels = []  # List to store labels

    for wav_file in test_wav_files:
        # Preprocess the audio for prediction
        audio_slices = preprocess_audio(wav_file)

        # Make predictions using the specified threshold
        yhat, predictions = predict_audio(audio_slices, args.val)

        # Output the predictions
        print(f"Predictions for {wav_file}:")
        print("yhat:", yhat)
        #print("yhat:", [[round(val[0], 4)] for val in yhat])
        print("Predictions:", predictions)
        
        # Convert yhat NumPy array to a list and extend the list
        yhat_values.extend(yhat[:, 0])
        # Extract the label from the file name and add it to the labels list
        label = os.path.basename(wav_file).replace(".wav", "")
        labels.append(label)


    data = pd.DataFrame({'Labels': labels, 'yhat Values': yhat_values})

    # Create the interactive bar chart with color mapping
    fig = px.bar(data, x='Labels', y='yhat Values', color='yhat Values',
                color_continuous_scale='RdYlGn',  # Color scale from red to green
                labels={'yhat Values': 'yhat Values'},
                title='yhat Values for Test Samples',
                text='Labels')  # Show labels on the bars

    # Customize the x-axis labels
    fig.update_xaxes(tickangle=45)
    
    # Show the interactive plot
    fig.show()

    # Set the y-axis range from 0 to 1
    fig.update_yaxes(range=[0, 1])
    # Show the interactive plot
    fig.show()