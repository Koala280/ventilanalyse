import torch
import torch.nn as nn
import torch.optim as optim

class ConvRecurrentModel(nn.Module):
    def __init__(self, nb_filters1, nb_filters2, nb_filters3, nb_filters4, nb_filters5,
                 ksize, pool_size_1, pool_size_2, pool_size_3, lstm_count, dense_size1, num_classes):
        super(ConvRecurrentModel, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=nb_filters1, kernel_size=ksize, stride=1,padding=0)  # Modify the in_channels according to your input shape
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=pool_size_1)
        self.conv_2 = nn.Conv2d(in_channels=nb_filters1, out_channels=nb_filters2, kernel_size=ksize, stride=1,padding=0)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=pool_size_1)
        self.conv_3 = nn.Conv2d(in_channels=nb_filters2, out_channels=nb_filters3, kernel_size=ksize, stride=1,padding=0)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=pool_size_1)
        self.conv_4 = nn.Conv2d(in_channels=nb_filters3, out_channels=nb_filters4, kernel_size=ksize, stride=1,padding=0)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=pool_size_2)
        self.conv_5 = nn.Conv2d(in_channels=nb_filters4, out_channels=nb_filters5, kernel_size=ksize, stride=1,padding=0)
        self.maxpool_5 = nn.MaxPool2d(kernel_size=pool_size_2)

        self.flatten = nn.Flatten()
        self.pool_lstm = nn.MaxPool2d(kernel_size=pool_size_3)

        self.bidirectional_gru = nn.GRU(input_size=nb_filters1, hidden_size=lstm_count, bidirectional=True)
        self.concat = nn.Linear(nb_filters5 + 2 * lstm_count, dense_size1)
        self.output = nn.Linear(dense_size1, num_classes)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_1 = self.relu(conv_1)
        pool_1 = self.maxpool_1(conv_1)

        conv_2 = self.conv_2(pool_1)
        conv_2 = self.relu(conv_2)
        pool_2 = self.maxpool_2(conv_2)

        conv_3 = self.conv_3(pool_2)
        conv_3 = self.relu(conv_3)
        pool_3 = self.maxpool_3(conv_3)

        conv_4 = self.conv_4(pool_3)
        conv_4 = self.relu(conv_4)
        pool_4 = self.maxpool_4(conv_4)

        conv_5 = self.conv_5(pool_4)
        conv_5 = self.relu(conv_5)
        pool_5 = self.maxpool_5(conv_5)

        flatten1 = self.flatten(pool_5)

        pool_lstm1 = self.pool_lstm(x)
        squeezed = pool_lstm1.squeeze(-1)

        lstm_output, _ = self.bidirectional_gru(squeezed)
        lstm_output = lstm_output[:, -1, :]

        concat = torch.cat((flatten1, lstm_output), dim=1)
        concat = self.concat(concat)

        output = self.output(concat)

        return output