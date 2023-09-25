# %%
import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio
from split_audio_by_duration import split_audio_by_duration 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import datetime

""" 
POSITIVE: 1
NEGATIVE: 0
"""

SAVE_MODEL = True

# %%
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


# %%
POS = os.path.join('./', 'audios', 'positive')
NEG = os.path.join('./', 'audios', 'negative')


# %%
pos = tf.data.Dataset.list_files(POS+'\*.wav')
neg = tf.data.Dataset.list_files(NEG+'\*.wav')


# %%
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)


# %%
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


# %%
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

# %%
train = data.take(int(len(data)*.8))
test = data.skip(int(len(data)*.8)).take(int(len(data)-len(data)*.8))

# %%
# %%
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(491, 257,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# %%
hist = model.fit(train, epochs=4, validation_data=test)

if SAVE_MODEL:
    model.save(f"audio_classification_model{datetime.datetime.now().strftime('%Y.%m.%d.%H%M')}.h5")

"""
# %%
X_test, y_test = test.as_numpy_iterator().next()


# %%
yhat = model.predict(X_test)

# %%
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

# %%
print(yhat)
print(y_test.astype(int))

# %%
#TODO wav_path als parameter
wav_path = os.path.join('./', 'audios', 'test', 'rec2.wav')
#wav_path = os.path.join('./', 'audios', 'test', 'rec6.wav')
#wav_path = os.path.join('./', 'audios', 'test', 'rec8.wav')

# %%
wav = load_wav_16k_mono(wav_path)

# %%
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)

# %%
samples, index = audio_slices.as_numpy_iterator().next()

# %%
def preprocess_prediction(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([16000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# %%
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_prediction)
audio_slices = audio_slices.batch(64)

#TODO Hier Modell Laden vllt auch als parameter
#model = tf.load("model.h5")


# %%
#TODO Schwellenwert als parameter
SCHWELLENWERT = 0.52
yhat = model.predict(audio_slices)
yhat = [1 if prediction > SCHWELLENWERT else 0 for prediction in yhat]
print("prediction:", yhat)
"""

# %%
#print("Funktionierende audios erkannt:", broken_air)