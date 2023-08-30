import torch.nn as nn

class CNNNetworkTutorial(nn.Module):
    def __init__(self,
                 hidden1 = 5,
                 hidden2 = 5,
                 hidden3 = 5,
                 hidden4 = 5,
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
        #self.linear = nn.Linear(self.hidden2 * 5 * 4, num_classes)
        # in case dimensions don't work
        self.linear = nn.Linear(self.hidden2 * 5 * 4 * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        prediction = self.sigmoid(logits)

        return prediction