from pydub import AudioSegment
import os

def _get_new_audio_id(folder_path):
    # Get all file names in the folder
    files = os.listdir(folder_path)

    # Extract numbers from file names
    numbers = []
    for file in files:
        if file.endswith(".wav"):
            name = os.path.splitext(file)[0]
            try:
                number = int(name)
                numbers.append(number)
            except ValueError:
                pass

    # Find the highest number
    highest_number = max(numbers) if numbers else -1

    # Increment the highest number by one
    new_number = highest_number + 1

    return new_number




def split_audio_by_duration(audio_path, segment_duration = 1000):
    output_folder, _ = os.path.split(audio_path)
    audio_path = audio_path.replace("/", "\\\\")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_sample_width(2)  # Set sample width to 2 bytes (16 bits)
    audio = audio.set_frame_rate(16000)  # Set sample rate to 16000 Hz
    audio_duration = len(audio)
    segment_start = 0
    segment_end = segment_duration
    
    segments = []
    while segment_end <= audio_duration:
        segment = audio[segment_start:segment_end]
        segments.append(segment)
        segment_start += segment_duration
        segment_end += segment_duration
        output_path = os.path.join(output_folder, f"{_get_new_audio_id(output_folder)}.wav")
        segment.export(output_path, format="wav")
    
    # Remove the original audio file
    os.remove(audio_path)

if __name__ == "__main__":
    classes = {'positive': 0, 'negative': 1}
    for label in classes.keys():
        for file in os.listdir("./audios/" + label):
            file_path = "./audios/" + label + "/" + file
            split_audio_by_duration(file_path)