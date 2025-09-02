import os
import librosa

class AsrAudioSource: 
    sample_rate = 16000

    def __init__(self, wav_path):
        self.audio_array, _ = librosa.load(wav_path, sr=self.sample_rate)

