import numpy as np 
import torch 
torch.manual_seed(0)
import torch.nn as nn 
dtype = torch.float32


class Trainer(nn.Module):
    """
    Class to enable model training
        1. train_one_epoch(): method to carry out the training process for the model for a single epoch

        2. train_all_epochs(): method to carry out the training process for all the epochs
    """
    
    
    def __init__(self, network, num_epochs, learning_rate):
        super().__init__()
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = network 
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)


    def __str__(self):
        return f" Number of epochs = {self.num_epochs}\n Learning Rate = {self.lr}\n Loss Function = {self.loss}\n Optimizer = {self.optim}"

    
    
    def train_one_epoch(self, loader, epoch):
        self.net.train(True)
        running_loss = 0.0

        num_batches = len(loader)
        for idx, data in enumerate(loader):
            image_batch, label_batch = data
            
            self.optim.zero_grad()
            output = self.net(image_batch)

            loss = self.loss(output, label_batch)
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            
            # reporting for one epoch
            if (idx + 1) % num_batches == 0 :
                
                # calculating average loss for all the batches, and reporting it once per epoch.

                avg_loss = running_loss/num_batches
                print(f"Training: Epoch {epoch+1}, Batch = {idx+1}/{num_batches},   Training Loss: {avg_loss}")

        return avg_loss

            
            
        


    def train_all_epochs(self, loader):
        training_loss = [] 
        
        for e in range(self.num_epochs):
            
            # training the dataset for one epoch 
            epoch_loss = self.train_one_epoch(loader, epoch=e)
            training_loss.append(epoch_loss)


        return training_loss  








if __name__ == "__main__":
    testnetwork = nn.Sequential(
        nn.Linear(100, 50, dtype=dtype),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.LogSoftmax()
    )

    train = Trainer(network=testnetwork, num_epochs=100, learning_rate=0.001)
    train.train_one_epoch









