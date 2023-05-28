# Deep Learning framework
import torch
import torch.nn as nn

class CRNNNetwork(nn.Module):
    def __init__(self,
                 hidden1 = 16,
                 hidden2 = 32,
                 hidden3 = 64,
                 hidden4 = 64,
                 hidden5 = 64,
                 lstm_count = 64,
                 L2_regularization = 0.001,
                 dense_size1 = 256,
                 num_classes = 2):
        super(CRNNNetwork, self).__init__()
        self.input = 1 #mono
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.lstm_count = lstm_count
        self.dense_size1 = dense_size1
        self.L2_regularization = L2_regularization
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

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden4,
                out_channels=self.hidden5,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.pool_lstm = nn.MaxPool2d(kernel_size=3)

        self.bidirectional_gru = nn.GRU(input_size=self.input, hidden_size=self.lstm_count, bidirectional=True)
        self.concat = nn.Linear(self.hidden5 + 2 * self.lstm_count, self.dense_size1)
        self.output = nn.Linear(self.dense_size1, self.num_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        flatten1 = self.flatten(conv5)

        pool_lstm = self.pool_lstm(x)
        squeezed = pool_lstm.squeeze(-1)
        
        lstm_output, _ = self.bidirectional_gru(squeezed)
        lstm_output = lstm_output[:, -1, :]

        concat = torch.cat((flatten1, lstm_output), dim=1)
        concat = self.concat(concat)

        output = self.output(concat)

        return output
