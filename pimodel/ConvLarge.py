import torch.nn as nn


class ConvLarge(nn.Module):
    def __init__(self, num_classes):
        super(ConvLarge, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(0.5),

            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, 1, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.linear = nn.Linear(128, num_classes)

    def forward(self, x, return_feature=False):
        f = self.conv_layer(x)
        x = self.linear(f.squeeze())
        if return_feature:
            return [x, f.squeeze()]
        else:
            return x
