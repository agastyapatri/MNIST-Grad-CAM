import numpy as np 
import os 
import torch
torch.manual_seed(0)
import torch.nn as nn 


# Custom Imports
from src.loadvis import Loader
from src.loadvis import Visualizer


if __name__ == "__main__":
    loader = Loader(PATH="/home/agastya123/PycharmProjects/DeepLearning/MNIST-Grad-CAM/data/", num_batches=64)

    #1. Loading the data 
    trainloader, testloader = loader.getdata()
    
    #2. Visualizing the data
    visualizer = Visualizer()
    # visualizer.visualize(trainloader)


    #3. Building the network 
    
    





