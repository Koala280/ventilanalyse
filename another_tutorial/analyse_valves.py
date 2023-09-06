"""
    The .tflite file of the model must be in the same file directorey on the Raspberry Pi
"""

# Processing
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import scipy.signal
# Inference
from tflite_runtime.interpreter import Interpreter
# CLI
from argparse import ArgumentParser, ArgumentTypeError


"""
CLI handling
----------------------------------------------------------------------
"""

# Type for checking whether a float value is in the specified range    
def float_range(minimum, maximum):
    def float_range_checker(arg):
        try:
            value = float(arg)
        except ValueError:    
            raise ArgumentTypeError("must be a floating point number")
        if value < minimum or value > maximum:
            raise ArgumentTypeError("must be in range [" + str(minimum) + " .. " + str(maximum)+"]")
        return value
    return float_range_checker    

parser = ArgumentParser()

parser.add_argument('-t','--threshold', help='Define the threshold for the prediction values (default %(default)s)', type=float_range(0, 1), metavar=['0-1'], default=0.52)
parser.add_argument('-v','--valve', help='Set the valve time in ms (default %(default)s)', type=int, default=200)
parser.add_argument('-c','--cycle', help='Set the cycle duration in ms (default %(default)s)', type=int, default=250)
parser.add_argument('-v', '--verbose', help='Prints verbose output', action='store_true')

args = parser.parse_args()

if args.valve >= args.cycle:
    raise ArgumentTypeError('valve time must be smaller than cycle duration')


"""
Parameters and variables
----------------------------------------------------------------------
"""
# Values for the CLI arguments
prediction_threshold = args.threshold
valve_time = args.valve
cycle_duration = args.cycle

# Calculate pause between each valve operation
valve_pause = cycle_duration - valve_time

# Parameters
rec_duration = 0.5
sample_rate = 16000
resample_rate = 16000
model_path = 'audio_classification_lite.tflite'

"""
ML model
----------------------------------------------------------------------
"""

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

"""
Defining functions and methods
----------------------------------------------------------------------
"""

# convert to mono and resample
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


wav = load_wav_16k_mono(wav_path)


# Build Function to Convert Clips into Windowed Spectrograms
def preprocess_2(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([16000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# Convert Longer Clips into Windows and Make Predictions
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_2)
audio_slices = audio_slices.batch(64)

yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.52 else 0 for prediction in yhat]
print(yhat)


# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    GPIO.output(led_pin, GPIO.LOW)

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    mfccs = python_speech_features.base.mfcc(window, 
                                        samplerate=new_fs,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=num_mfcc,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)
    mfccs = mfccs.transpose()

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]
    if val > word_threshold:
        print('stop')
        GPIO.output(led_pin, GPIO.HIGH)

    if debug_acc:
        print(val)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass