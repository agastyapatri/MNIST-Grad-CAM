import numpy as np 
import torch 
import torch.nn as nn 
dtype = torch.float32


class GradCAM(nn.Module):
    """
    Gradient Weighted Class Activation Mapping on a classification model.
    """

    def __init__(self, network, class_idx):
        super().__init__()
        self.net = network 
        self.target_class = class_idx 
        self.classes = range(10)

    def activation_map(self, layer):
        """
        Function to find activation maps of the Desired Convolution Layer
        :return A: The activation map of the desired convolution layer
        """
        print(layer)


    def gradient(self):
        """
        Function to calculate the gradient of the class score of the desired class wrt the activation map 
        """
        pass 

    def alpha_ck(self):
        """
        Function to calculate the neuron importance weights by global average pooling the gradients
        """
        pass 
    
    def map(self):
        """
        Function to calculate the GradCAM scores for the class, and return a map which can be plotted.
        """
        pass 


if __name__ == "__main__":


    CONV1 = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
    
    CONV2 = nn.Sequential(
            # Conv Layer 2 
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ) 

    FCLayers = nn.Sequential(
            # Fully Connected Layer 1
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=96, out_features=48, dtype=dtype),
            nn.Sigmoid(),

            # Fully Connected Layer 2
            nn.Linear(in_features=48, out_features=10, dtype=dtype),
            nn.LogSoftmax(dim=1)
        )

    CNN = nn.Sequential(
        CONV1,
        CONV2,
        FCLayers
    )

    
    testdata = torch.randn(64, 1,28, 28, dtype=dtype, requires_grad=True)

    print(CNN[0][0].weight[0])



    
    