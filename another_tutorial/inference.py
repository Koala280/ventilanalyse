import os
import argparse
import tensorflow as tf
import tensorflow_io as tfio

""" 
POSITIVE: 1
NEGATIVE: 0
"""

MODEL = "audio_classification_model20230913133824.h5"
TEST_WAV = "rec200.wav"

# Load the saved model
model = tf.keras.models.load_model("./another_tutorial/"+MODEL)

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

# Function to preprocess audio for prediction
def preprocess_audio(audio_path, silence_threshold=0.01, segment_length=50):
    # Load and preprocess the audio file
    wav = load_wav_16k_mono(audio_path)  # You can reuse your `load_wav_16k_mono` function

    # Find the first silence point
    silence_mask = tf.math.less(tf.abs(wav), silence_threshold)
    first_silence_idx = tf.argmax(tf.cast(silence_mask, tf.int32))

    # Determine the start and end indices for the audio to be included before the first silence point
    start_idx = max(0, first_silence_idx - int(segment_length/1000 * 16000))
    end_idx = first_silence_idx

    # Include the audio before the first silence point
    audio_before_silence = wav[start_idx:end_idx]

    # Split the audio into segments starting from the first silence point
    segments = []
    current_idx = first_silence_idx
    while current_idx < len(wav):
        start_idx = current_idx
        end_idx = current_idx + int(segment_length/1000 * 16000)  # Split into 'segment_length' seconds
        if end_idx > len(wav):
            end_idx = len(wav)
        segment = wav[start_idx:end_idx]
        if len(segment) == int(segment_length/1000 * 16000):  # Only consider segments of fixed length
            segments.append(segment)
        current_idx = end_idx

    audio_slices = tf.convert_to_tensor([audio_before_silence] + segments)

    audio_slices = tf.keras.utils.timeseries_dataset_from_array(
        audio_slices, audio_slices, sequence_length=16000, sequence_stride=16000, batch_size=1
    )
    audio_slices = audio_slices.map(preprocess_prediction)
    audio_slices = audio_slices.batch(64)
    return audio_slices

# Function to make predictions on audio
def predict_audio(audio_slices, threshold=0.5):
    yhat = model.predict(audio_slices)
    predictions = [1 if prediction > threshold else 0 for prediction in yhat]
    return yhat, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification Inference")
    parser.add_argument("--wav", type=str, default="./another_tutorial/"+TEST_WAV, help="Path to the audio file to classify")
    parser.add_argument(
        "--val",
        type=float,
        default=0.5,
        help="Threshold for classification (default: 0.5)",
    )
    args = parser.parse_args()

    # Preprocess the audio for prediction
    audio_slices = preprocess_audio(args.wav)

    # Make predictions using the specified threshold
    yhat, predictions = predict_audio(audio_slices, threshold=args.val)

    # Output the predictions
    print("yhat:", [[round(val[0], 6)] for val in yhat])
    print("Predictions:", predictions)
