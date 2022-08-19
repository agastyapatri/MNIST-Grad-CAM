import torch 
import numpy as np
import torch.nn as nn 
dtype = torch.float16


class ConvolutionalNetwork(nn.Module):
    """
    Defining the model that classifies the images
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"ConvNet structure: {self.network()}"
        

    def network(self):
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=None, dtype=dtype),

            
            nn.ReLU(),


        )
        return cnn 
        
        

if __name__ == "__main__":
    model = ConvolutionalNetwork()
    print(model)