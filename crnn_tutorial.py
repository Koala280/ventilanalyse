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

from scripts.dataset import AudioDataset

AUDIO_DIR = "audios/labeled/"

dict_genres = {'positive': 0, 'negative': 1, 'noise': 2}

reverse_map = {v: k for k, v in dict_genres.items()}

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(DEVICE)

class CNNNetworkTutorial(nn.Module):
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

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimizer):
    for input, target in data_loader:
        input, target = input.to(DEVICE), target.to(DEVICE)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimizer)
        print("---------------------------")
    print("Finished training")


data = []

for label in dict_genres.keys():
    for folder in os.listdir(AUDIO_DIR + label):
        for file in os.listdir(AUDIO_DIR + label + "/" + folder):
            file_path = AUDIO_DIR + label + "/" + folder + "/" + file
            positive = int(label == "positive")
            negative = int(label == "negative")
            noise = int(label == "noise")
            data.append((file_path, positive,
                        negative))

file_path, positive, negative = zip(*data)
df = pd.DataFrame({"file_path": file_path, "positive": positive, "negative": negative, "noise": noise})

print(df.head(5))

BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 0.001
SAVE = False

dataset = AudioDataset(df)

train_dataloader = create_data_loader(dataset, BATCH_SIZE)

# construct model and assign it to device
model = CNNNetworkTutorial().to(DEVICE)
summary(model.cuda(), (1, 64, 44))

# initialise loss funtion + optimiser
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

# train model
train(model, train_dataloader, loss_fn, optimizer, EPOCHS)

# save model
if SAVE:
    torch.save(model.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")