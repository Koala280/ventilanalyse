# Deep Learning framework
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# Audio processing
import torchaudio
import torchaudio.transforms as T
import librosa

# Pre-trained image models
# import timm

# Play the audio in Jupyter notebook
from IPython.display import Audio
import pandas as pd
import os
import numpy as np

AUDIO_DIR = "audios/labeled/"

dict_genres = {'positive': 0, 'negative': 1, 'noise': 2}

reverse_map = {v: k for k, v in dict_genres.items()}
print(reverse_map)

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(DEVICE)

class CRNNNetworkTutorial(nn.Module):
    def __init__(self,
                 hidden1 = 16,
                 hidden2 = 32,
                 hidden3 = 64,
                 hidden4 = 128,
                 num_classes = 2):
        super().__init__()
        self.input = 1 #mono
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input,
                out_channels=self.hidden1,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden1,
                out_channels=self.hidden2,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden2,
                out_channels=self.hidden3,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden3,
                out_channels=self.hidden4,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.hidden4 * 5 * 4, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.to(DEVICE)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        prediction = self.softmax(logits)

        return prediction
