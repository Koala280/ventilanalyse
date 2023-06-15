import torch
import torchaudio

from models.cnn_tutorial import CNNNetworkTutorial
from scripts.dataset import AudioDataset
from scripts.dataframe import get_dataframe


class_mapping = [
    "positive",
    "negative"
]


def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetworkTutorial()
    state_dict = torch.load("cnn_tutorial.pth")
    cnn.load_state_dict(state_dict)
    df = get_dataframe()
    ds = AudioDataset(df, target_sample_rate=22050)


    # get a sample from the dataset for inference
    input, target = ds[0][0], ds[0][1].argmax(0) # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target)
    print(f"Predicted: '{predicted}', expected: '{expected}'")