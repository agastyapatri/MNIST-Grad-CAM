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

    def __init__(self, network, class_idx):

        super().__init__()
        self.net = network 
        self.target_class = class_idx 
        self.classes = range(10)
        self.gradients = None 

        # breaking down the network into usable chunks. 
        self.CONV = nn.Sequential().append(self.net[0]).append(self.net[1][:2])
        self.maxpool = nn.MaxPool2d(kernel_size=2) 
        self.FC = self.net[2]
        
    


    def forward(self, input_tensor):
        # Getting output up to the ReLU() of CONV2
        output = self.CONV(input_tensor)

        # registering the hook to the last activation map
        output.register_backward_hook(lambda grad : )

        # Maxpooling the output from ReLU() of CONV2
        output = self.maxpool(output)

        # Getting the output of the Fully Connected Layers
        output = self.FC(output)
        return output


    def extract_gradients(self):
        # There should be 24 gradients
        return self.gradients


    def activations(self, input):
        # 24 feature maps of 4x4 kernel
        activation_map = self.CONV(input)
        return activation_map 



    def L_c(self, image):

        y_c = self.net(image)
        A_ijk = self.activations(image)

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




    a = torch.ones(1, 1, 28, 28, dtype=dtype, requires_grad=True)
    b = torch.ones(1, 1, 28, 28, dtype=dtype, requires_grad=True)
    
    gradcamtest = GradCAM(network=CNN, class_idx=None)
    print(gradcamtest.extract_gradients())
    # preds = torch.argmax(gradcamtest(a))
    # print(gradcamtest.L_c(a))


    
    





    """TESTING STUFF"""
    x = torch.tensor(np.ones([1,10]), requires_grad=True, dtype=dtype)
    y = 2*x
    y.register_hook(lambda grad: grad+1)


  



    

    

    
    



    