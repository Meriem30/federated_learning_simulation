#!/usr/bin/env python3

import collections

import torch.nn as nn


class LeNet5(nn.Module):

    input_size = (224, 224)

    def __init__(self, input_channels=3, num_classes=1):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.convnet = nn.Sequential(
            collections.OrderedDict(
                [
                    ("c1", nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)),
                    ("relu1", nn.ReLU()),
                    ("s2", nn.MaxPool2d(kernel_size=2, stride=2)),  # (64, 112, 112)
                    
                    ("c3", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
                    ("relu3", nn.ReLU()),
                    ("s4", nn.MaxPool2d(kernel_size=2, stride=2)),  # (128, 56, 56)
                    
                    ("c5", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
                    ("relu5", nn.ReLU()),
                    ("s6", nn.MaxPool2d(kernel_size=2, stride=2)),  # (256, 28, 28)
                ]
            )
        )

        # Compute the correct input size for the first fully connected layer
        self.flattened_size = 256 * 28 * 28  # Adjusted based on pooling

        self.fc = nn.Sequential(
            collections.OrderedDict(
                [
                    ("f6", nn.Linear(self.flattened_size, 84)),  # Adjusted input size
                    ("relu6", nn.ReLU()),
                    ("f7", nn.Linear(84, self.num_classes)),
                ]
            )
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(x.size(0), -1)  # Flatten before fully connected layer
        output = self.fc(output)
        return output