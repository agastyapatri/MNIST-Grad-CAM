"""
Classes to load, visualize etc
"""
import numpy as np 
import torch 
torch.manual_seed(0)

import torch.nn as nn 
import matplotlib.pyplot as plt
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader




class Loader(nn.Module):
    """
    Class to load the MNIST data into PyTorch dataloaders
        1. getdata(): Function to fetch the MNIST data and store into dataloaders.
    """
    
    
    
    def __init__(self, PATH, num_batches):
        super().__init__()
        self.path = PATH 
        self.num_batches = num_batches
        
    def __str__(self):
        return f"Object of class Loader"

    def getdata(self):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, ), (1, ))
        ])

        trainset = datasets.MNIST(root=self.path, train=True, download=True, transform=trans)
        testset = datasets.MNIST(root=self.path, train=False, download=True, transform=trans)



        trainloader = DataLoader(dataset=trainset, batch_size=self.num_batches, shuffle=True)

        testloader = DataLoader(dataset=testset, batch_size=self.num_batches, shuffle=True)

        print(len(trainloader))
        print(len(testloader))







class Visualizer:
    """
    Class to Visualize the images in MNIST
    """



if __name__ == "__main__":
    
    test_Loader = Loader(PATH="/home/agastya123/PycharmProjects/DeepLearning/MNIST-Grad-CAM/data/", num_batches=None)
    test_Loader.getdata()
    
