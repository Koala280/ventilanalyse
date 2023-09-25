"""
    The .tflite file of the model must be in the same file directory on the Raspberry Pi
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
Defining functions and methods
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

# Convert to mono and resample
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

# Build function to convert clips into windowed spectrograms
def preprocess_2(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([16000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# Convert longer clips into windows and apply preprocessing
def preprocess_slices(wav):
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_2)
    audio_slices = audio_slices.batch(64)
    return audio_slices

# Will automatically run when this script is started from console
def main():
    
    """
    CLI handling
    ----------------------------------------------------------------------
    """  

    parser = ArgumentParser()

    parser.add_argument('-t','--threshold', help='Define the threshold for the prediction values (default: %(default)s)', type=float_range(0, 1), metavar=['0-1'], default=0.52)
    parser.add_argument('-v','--valve', help='Set the valve time in ms (default: %(default)s)', type=int, default=200)
    parser.add_argument('-c','--cycle', help='Set the cycle duration in ms (default: %(default)s)', type=int, default=250)
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
    wav_path = 'valve_test.wav'

    """
    ML model
    ----------------------------------------------------------------------
    """

    # Load model (interpreter)
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Do preprocessing of audio file
    wav = load_wav_16k_mono(wav_path)
    audio_slices = preprocess_slices(wav)

    # Set interpreter input
    input_data = audio_slices
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference on input
    print("running inference...")
    interpreter.invoke()

    # Output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    yhat = output_data[0]

    # Apply threshold to convert confidence score to classes
    yhat = [1 if prediction > prediction_threshold else 0 for prediction in yhat]
    print(yhat)
    

if __name__ == "__main__":
    main()