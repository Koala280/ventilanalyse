# Deep Learning framework
import torch
from torch.utils.data import Dataset, DataLoader

# Audio processing
import torchaudio
import torchaudio.transforms as T

class AudioDataset(Dataset):
    def __init__(self, 
                df,
                audio_length = 1, #max length of negative sample = 7.1 seconds
                target_sample_rate=22050):
        self.df = df
        self.file_paths = df['file_path'].values
        self.labels = df[['positive', 'negative']].values
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * audio_length

        self.melspectrogram = T.MelSpectrogram(sample_rate=self.target_sample_rate,
                                               n_mels=128,
                                               n_fft=2048,
                                               hop_length=512)#.to(DEVICE)
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        # get audio path and label
        audio_path = self.file_paths[index]
        label = self.labels[index]

        # Load audio from file to waveform
        audio, sample_rate = torchaudio.load(audio_path)

        # register to cpu/gpu
        audio = audio#.to(DEVICE)

        # Resample to target sample rate
        audio = self._resample(audio, sample_rate)

        # Convert to mono
        audio = self._mono(audio)

        # Adjust number of samples
        audio = self._crop(audio)
        audio = self._right_pad(audio)

        # Add any preprocessing you like here 
        # (e.g., noise removal, etc.)
        
        
        # Add any data augmentations for waveform you like here
        # (e.g., noise injection, shifting time, changing speed and pitch)
        """ 
        wave_transforms = T.PitchShift(sample_rate, 4)
        audio = wave_transforms(audio)
        """

        # Convert to Mel spectrogram
        melspec = self.melspectrogram(audio)
        
        # Add any data augmentations for spectrogram you like here
        # (e.g., Mixup, cutmix, time masking, frequency masking)
        """ 
        spec_transforms = T.FrequencyMasking(freq_mask_param=80)
        melspec = spec_transforms(melspec)
        """
        return melspec, label

    def _crop(self, audio):
        if audio.shape[1] > self.num_samples:
            audio = audio[:, :self.num_samples]
        return audio
    
    def _right_pad(self, audio):
        audio_length = audio.shape[1]
        if audio_length < self.num_samples:
            num_missing_samples = self.num_samples - audio_length
            last_dim_padding = (0, num_missing_samples)
            audio = F.pad(audio, last_dim_padding)
        return audio

    def _resample(self, audio, sr):
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        return audio

    def _mono(self, audio):
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio