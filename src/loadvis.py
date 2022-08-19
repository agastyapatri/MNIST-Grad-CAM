"""
Classes to load, visualize etc
"""
from pydoc import visiblename
from socket import TIPC_DEST_DROPPABLE
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
        return f"Trainloader: length = {len(self.getdata()[0])}  | Testloader: length = {len(self.getdata()[1])}\nBatch Size = {self.num_batches}"

    def getdata(self):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, ), (1, ))
        ])

        trainset = datasets.MNIST(root=self.path, train=True, download=True, transform=trans)
        testset = datasets.MNIST(root=self.path, train=False, download=True, transform=trans)



        trainloader = DataLoader(dataset=trainset, batch_size=self.num_batches, shuffle=True)

        testloader = DataLoader(dataset=testset, batch_size=self.num_batches, shuffle=True)

        return trainloader, testloader






class Visualizer:
    """
    Class to Visualize the images in MNIST
        1. visualize(): method to enable the plotting of the digits. 
        2. distribution(): method to visualize the class distribution of the dataset. 
        3. 
    """
    def __init__(self):
        pass

    def visualize(self, loader):
        test =  list(enumerate(loader))
        
        # For the first batch 
        image_batch = test[0][1][0]
        label_batch = test[0][1][1]

        image = image_batch[0].view(28, 28)
        label = label_batch[0]

        plt.title(f"Image of the digit:{label.item()}")
        plt.imshow(image.numpy(), cmap="Blues")
        plt.show()


    def distribution(self):
        pass 


if __name__ == "__main__":
    
    loader = Loader(PATH="/home/agastya123/PycharmProjects/DeepLearning/MNIST-Grad-CAM/data/", num_batches=64)
    trainloader, testloader = loader.getdata()



    



    








    
