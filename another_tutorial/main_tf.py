# %%
import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio
from split_audio_by_duration import split_audio_by_duration 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten



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

# %%
X_test, y_test = test.as_numpy_iterator().next()
print(X_test.shape)
print(y_test.shape)


# %%
yhat = model.predict(X_test)
yhat[:3]

# %% [markdown]
# ## 8.2 Convert Logits to Classes 

# %%
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
yhat

# %%
print(tf.math.reduce_sum(yhat))
print(tf.math.reduce_sum(y_test))

# %%
print(yhat)
print(y_test.astype(int))

# %% [markdown]
# # 9. Build Forest Parsing Functions

# %% [markdown]
# ## 9.1 Load up MP3s

# %%
wav_path = os.path.join('./', 'audios', 'test', 'rec2.wav')
#wav_path = os.path.join('./', 'audios', 'test', 'rec2.wav')
#wav_path = os.path.join('./', 'audios', 'test', 'rec8.wav')

# %%
wav = load_wav_16k_mono(wav_path)
wav.shape

# %%
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)

# %%
samples, index = audio_slices.as_numpy_iterator().next()

# %%
len(audio_slices)

# %%
samples.shape

# %% [markdown]
# ## 9.2 Build Function to Convert Clips into Windowed Spectrograms

# %%
def preprocess_2(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([16000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# %% [markdown]
# ## 9.3 Convert Longer Clips into Windows and Make Predictions

# %%
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_2)
audio_slices = audio_slices.batch(64)

# %%
yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.52 else 0 for prediction in yhat]
yhat

# %% [markdown]
# ## 9.4 Group Consecutive Detections

# %%
from itertools import groupby

# %%
yhat = [key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()

# %%
calls

# %% [markdown]
# # 10. Make Predictions

# %% [markdown]
# ## 10.1 Loop over all recordings and make predictions

# %%
results = {}
for file in os.listdir(os.path.join('data', 'Forest Recordings')):
    FILEPATH = os.path.join('data','Forest Recordings', file)
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    results[file] = yhat

# %%
results

# %% [markdown]
# ## 10.2 Convert Predictions into Classes

# %%
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
class_preds

# %% [markdown]
# ## 10.3 Group Consecutive Detections

# %%
postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessed

# %% [markdown]
# # 11. Export Results

# %%
import csv

# %%
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])


