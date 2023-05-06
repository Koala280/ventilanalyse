from pydub import AudioSegment, silence
import os

# Name der wav-Datei
recName = "rec5"
# min_silence_duration muss bei anderen Aufnahmen für verschiedene Ventilzyklen angepasst werden

# Lade die WAV-Datei
audio_file = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), f"..\\..\\recordings\\{recName}.wav"))

# Definiere die Schwellenwerte für die Geräuschtrennung
min_silence_duration = 40  # Mindestlänge der Stille in ms
silence_threshold = -50     # Mindest-DB-Schwelle, die als "Stille" betrachtet wird

# Trenne die Einzeldateien
chunks = silence.split_on_silence(audio_file,
                                     min_silence_len=min_silence_duration,
                                     silence_thresh=silence_threshold,
                                     keep_silence=5)
for i, segment in enumerate(chunks):
    # Speichere die Datei nur, wenn sie eine Mindestlänge hat
    if len(segment) > min_silence_duration:
        # Speichere das Segment als WAV-Datei ab
            out_file = os.path.join(os.path.dirname(__file__), f"..\\..\\recordings\\{recName}_segments\\{recName}_segment_{i+1}.wav")
            segment.export(out_file, format="wav")
            print(f"Segment {i+1} von Datei {recName}.wav wurde gespeichert.")
