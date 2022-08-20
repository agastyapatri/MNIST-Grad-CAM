import numpy as np 
import torch 
import torch.nn as nn 
dtype = torch.float32


class GradCAM(nn.Module):
    """
    Class to perform Class Activation Mapping on a classification model 
    """

    def __init__(self, network):
        super().__init__()
        self.net = network 
        self.gradients = None 





if __name__ == "__main__":
    CNN = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            
            # Conv Layer 2 
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Fully Connected Layer 1
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=96, out_features=48, dtype=dtype),
            nn.Sigmoid(),

            # Fully Connected Layer 2
            nn.Linear(in_features=48, out_features=10, dtype=dtype),
            nn.LogSoftmax(dim=1)
        )
    
    testdata = torch.randn(64, 1,28, 28, dtype=dtype, requires_grad=True)
    
    
    res = CNN(testdata)
    print(res)


    
    