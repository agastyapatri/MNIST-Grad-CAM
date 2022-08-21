import numpy as np 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt  
dtype = torch.float32
torch.manual_seed(1)


class GradCAM(nn.Module):
    """
    Gradient Weighted Class Activation Mapping on a classification model.
        ~ forward() extracts the required gradients. 
        ~ activations(): getting activation map of final CONV Layer
        ~ extract_gradients(): getting the gradient of the class score wrt activation map
        ~ L_C(): finally computing the GradCAM scores
    """

    def __init__(self, network, class_idx, layer):

        super().__init__()
        self.net = network 
        self.target_class = class_idx 
        self.classes = range(10)
        self.int_grad = None 

        # breaking down the network into usable chunks. 
        self.CONV = nn.Sequential().append(self.net[0]).append(self.net[1][:2])
        self.maxpool = nn.MaxPool2d(kernel_size=2) 
        self.FC = self.net[2]
        
    
    def hook(self, gradients):
        self.int_grad = gradients 


    def forward(self, input_tensor):
        # Getting output up to the ReLU() of CONV2
        output = self.CONV(input_tensor)

        # registering the hook
        h = output.register_hook(self.hook)

        # Maxpooling the output from ReLU() of CONV2
        output = self.maxpool(output)

        # Getting the output of the Fully Connected Layers
        output = self.FC(output)

        return output


    def activations(self, input):
        # 24 feature maps of 4x4 kernel
        activation_map = self.CONV(input)
        return activation_map 


    def extract_gradients(self):
        # There should be 24 gradients
        return self.int_grad

    def L_c(self, img):

        # gradients = self.extract_gradients()

        # A_ijk = self.activations(img)
        # alpha_ck = None

        # gradcamscores = max(torch.sum(torch.matmul(alpha_ck*A_ijk)) , 0)
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
        # Composing the different parts of the network together.
        CONV1,
        CONV2,
        FCLayers
    )



    
    testdata = torch.randn(1, 1, 28, 28, dtype=dtype, requires_grad=True)
    
    gradcamtest = GradCAM(network=CNN, class_idx=None, layer=2)
    preds = torch.argmax(gradcamtest(testdata))
    # print(gradcamtest(testdata)[0][5])
    print(preds.item())
    

    
    



    