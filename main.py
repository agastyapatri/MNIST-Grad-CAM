import numpy as np 
import os 
import torch
torch.manual_seed(0)
import torch.nn as nn 


# Custom Imports
from src.loadvis import Loader
from src.loadvis import Visualizer
from src.models import ConvolutionalNetwork
from src.traintest import Trainer


if __name__ == "__main__":
    loader = Loader(PATH="/home/agastya123/PycharmProjects/DeepLearning/MNIST-Grad-CAM/data/", batch_size=64)

    #1. Loading the data 
    trainloader, testloader = loader.getdata()
    
    #2. Visualizing the data
    visualizer = Visualizer()
    # visualizer.visualize(trainloader)


    #3. Building the network 
    model = ConvolutionalNetwork()
    cnn = model.network()
    

    #4. Initial Training cycle and Getting baselines
    trainer = Trainer(network=cnn, num_epochs=10, learning_rate=0.01)
    trainer.train_one_epoch(trainloader, epoch=0)

    





    




    




