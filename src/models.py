import torch 
import numpy as np
import torch.nn as nn 
dtype = torch.float32


class ConvolutionalNetwork(nn.Module):
    """
    Defining the model that classifies the images
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"ConvNet structure: {self.network()}"
        

    def network(self):

        CONV = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3),

            # Conv Layer 2 
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU()

        )


        FC = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            # Fully Connected Layer 1
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=96, out_features=48, dtype=dtype),
            nn.Sigmoid(),

            # Fully Connected Layer 2
            nn.Linear(in_features=48, out_features=10, dtype=dtype),
            nn.LogSoftmax(dim=1)
        )


        CNN = nn.Sequential(
            CONV,
            FC
        )

        return CNN
        
        

if __name__ == "__main__":
    model = ConvolutionalNetwork()
    
    testdata = torch.randn(64, 1, 28, 28, dtype=dtype)
    mlp = model.network()

    