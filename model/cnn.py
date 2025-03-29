from torch import nn


class CNN(nn.Module):  # CNN: 2Conv, 2Pool
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 img_size: int = 28):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,  # In: (1, 28, 28)
                out_channels=16,  # Out: (16, 28, 28)
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Pooling: (16, 14, 14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # In: (16, 14, 14)
                out_channels=32,  # Out: (32, 14, 14)
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Pooling: (32, 7, 7)
        )

        self.output = nn.Linear(in_features=32*(img_size//4)*(img_size//4), out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # Out: (32, 7, 7)
        x = x.view(x.size(0), -1)  # batch_size: 32*7*7
        x = self.output(x)
        return x
