import numpy as np 
import os 
import torch
torch.manual_seed(0)
import torch.nn as nn 
import matplotlib.pyplot as plt 


# Custom Imports
from src.loadvis import Loader
from src.loadvis import Visualizer
from src.models import ConvolutionalNetwork
from src.traintest import TrainTest
from src.gradcam import GradCAM

if __name__ == "__main__":
    loader = Loader(PATH="/home/agastya123/PycharmProjects/DeepLearning/MNIST-Grad-CAM/data/", batch_size=64)

    #1. Loading the data 
    trainloader, testloader = loader.getdata()


    # Function to get a single image from the dataloaders based on its ground truth.
    def getanimage(loader, class_idx, plot=False):

        image_batch, label_batch  = next(iter(loader))

        #getting the desired from the first batch 
        for i in range(len(label_batch)):
            if label_batch[i] == class_idx:
                image_tensor = image_batch[i]

        if plot == True:
            plt.title(f"Image of the digit {class_idx}")
            plt.imshow(image_tensor[0], cmap="Greys")
            plt.show()

        return image_tensor

    # print(getanimage(loader=trainloader, class_idx=0, plot=False))



    
    #2. Visualizing the data
    visualizer = Visualizer()
    # visualizer.visualize(trainloader)


    #3. Building the network 
    model = ConvolutionalNetwork()
    cnn = model.network()
    

    #4. Initial Training cycle and Getting baselines
    traintest = TrainTest(network=cnn, num_epochs=10, learning_rate=0.01)

    # traintest.train_one_epoch(trainloader, epoch=0)
    trained_network, training_loss = traintest.train_all_epochs(trainloader)

    # 5. Testing the model on new data
    # traintest.test(network = trained_network, loader = testloader)

    # 6. Plotting the loss curve for the training cycle
    # visualizer.plotperf(training_loss, num_epochs = 10)


    # 7. Obtaining Grad-CAM for the model 

    gradcam = GradCAM(network=trained_network)
    data = next(iter(trainloader))
    print(gradcam.L_GC(image_data=data, class_name=3))

    


    





    




    




