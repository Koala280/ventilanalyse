"""
    The .wav file for the analysis and the .tflite file of the model must be in the same file directory on the Raspberry Pi
"""

# Audio cutting/preparation
from pydub import AudioSegment
import os
import shutil
import numpy as np
import math
# Audio processing
import tensorflow as tf
import tensorflow_io as tfio
# Inference
from tflite_runtime.interpreter import Interpreter
# CLI
from argparse import ArgumentParser, ArgumentTypeError


"""
Defining functions and methods
----------------------------------------------------------------------
"""
# Removes silence at the beginning of wav file
def remove_silence_start(rec_name, output_file, silence_threshold=-46):
    audio = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), f"..\\..\\recordings\\{rec_name}.wav"))
    print("Cutting silence at the end of the audio file")

    # Find index of first not silent sample
    start_index = next((i for i, x in enumerate(audio) if x.dBFS > silence_threshold), None)

    # Safety buffer for not cutting to much of the last valve sound
    safety_margin = 50

    if (start_index is not None) and (start_index > safety_margin):
        # Cut silence at the beginning of the audio
        print(f"Old audio length: {len(audio)} ms")
        audio = audio[start_index - safety_margin:]
        print(f"New audio length: {len(audio)} ms")

        # Save edited file
        audio.export(output_file, format="wav")
    else:
        print("No cutting required")

# Removes silence at the end of wav file
def remove_silence_end(rec_name, output_file, silence_threshold=-46):
    audio = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), f"..\\..\\recordings\\{rec_name}.wav"))
    print("Cutting silence at the end of the audio file")

    # Find index of last not silent sample
    end_index = next((len(audio) - 1 - i for i, x in enumerate(reversed(audio)) if x.dBFS > silence_threshold), None)

    # Safety buffer for not cutting to much of the last valve sound
    safety_buffer = 50

    if (end_index is not None) and ((end_index + safety_buffer) < (len(audio) - 1)):
        # Cut silence at the end of the audio
        print(f"Old audio length: {len(audio)} ms")
        audio = audio[:-(len(audio) - (end_index + safety_buffer) - 1)]
        print(f"New audio length: {len(audio)} ms")

        # Save edited file
        audio.export(output_file, format="wav")
    else:
        print("No cutting required")

# Determine the k-th percentile based on the proportion of valve noise in the audio file
def get_percentile_dbfs(rec_name, valve_time, cycle_duration):
    audio = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), f"..\\..\\recordings\\{rec_name}.wav"))

    # Takes a sample out of the middle of the audio file for the percentile determination -> beginning/end of audio file could contain silence
    samples = []
    for i, x in enumerate(audio[len(audio)/2 - 4*(cycle_duration):len(audio)/2]):
        samples.append(x.dBFS)

    # Determines percentile
    noise_percentage = 100 - (100 * (valve_time / cycle_duration))
    percentile_dbfs = np.percentile(samples, noise_percentage)
    percentile_dbfs = math.trunc(percentile_dbfs)
    print(f"{noise_percentage}% of the audio file are quieter than {percentile_dbfs} dBFS")

    return percentile_dbfs

# Cut WAV file in equally long chunks
def split_wav(rec_name, output_directory, valve_time, cycle_duration):
    audio = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), f"..\\..\\recordings\\{rec_name}.wav"))
    print("Splitting wav file into single valve sounds")

    # Make sure that the output directory exists
    # Or delete old directory from previous script execution
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        os.rmdir(output_directory)
        shutil.rmtree(output_directory)

        

    total_duration = len(audio)
    chunk_number = 1
    current_position = 0

    while current_position < total_duration:
        end_position = current_position + cycle_duration
        if end_position > total_duration:
            end_position = total_duration

        chunk = audio[current_position:end_position]
        current_position = end_position

        chunk_length = len(chunk)
        # Fill last audio chunk with zeroes, if it's too short
        if chunk_length < cycle_duration:
            if chunk_length <= valve_time:
                break
            else:
                chunk += AudioSegment.silent(duration=(cycle_duration - len(chunk)), frame_rate=audio.frame_rate)

        print(f"New audio length: {len(audio)} ms")
        output_file = os.path.join(output_directory, f"{rec_name}_chunk_{chunk_number}.wav")
        chunk.export(output_file, format="wav")
        chunk_number += 1
    
    print(f"{chunk_number} valve sounds detected")

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
def load_16k_mono_wav(filename):
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
def preprocess_with_padding(sample, index):
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
    audio_slices = audio_slices.map(preprocess_with_padding)
    audio_slices = audio_slices.batch(64)
    return audio_slices

# Load audio chunks from directory and apply preprocessing
def preprocess_chunks(wav_chunks_dir):
    audio_chunks = tf.keras.utils.audio_dataset_from_directory(
                                wav_chunks_dir,
                                labels=None,
                                label_mode=None,
                                batch_size=1,
                                sampling_rate=16000,
                                output_sequence_length=16000,
                                shuffle=False)
    audio_chunks = audio_chunks.map(preprocess_with_padding)
    audio_chunks = audio_chunks.batch(128)
    return audio_chunks

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
    wav_chunks_dir = 'valve_test_chunks'

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
    wav = load_16k_mono_wav(wav_path)
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