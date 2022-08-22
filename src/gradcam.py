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
        self.class_idx = class_idx 
        self.classes = range(10)
        self.gradients = None 

        # breaking down the network into usable chunks. 
        self.CONV = self.net[0]
        self.FC = self.net[1] 




    def L_GC(self, img, class_idx):
        features = self.CONV(img)

        # batches, num_feature_maps, height_of_feature_map, width_of_feature_map
        _, N, H, W = features.size()
        outputs = self.FC(features)
        c_score = outputs[0, class_idx]

        # gradients of output logits with respect to last feature maps 
        grads = torch.autograd.grad(c_score, features)[0][0]

        # finding neuron importance weights by averaging the height + weight
        alphas = grads.mean(dim=-1).mean(dim=-1)


        # Grad-CAM Scores by summing + applying relu
        ReLU = lambda x: np.maximum(x, 0)
        
        # Grad_CAM Scores
        features = torch.reshape(features, shape=(N, H*W))
        L_c_intermediate = torch.matmul(alphas, features).detach().numpy()
        L_c_intermediate = L_c_intermediate.reshape(H, W) 
        L_c = np.maximum(L_c_intermediate, 0)

        return L_c






if __name__ == "__main__":


    CONV = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        
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
        CONV,
        FCLayers
    )




    a = torch.ones(1, 1, 28, 28, dtype=dtype, requires_grad=True)
    b = torch.ones(1, 1, 28, 28, dtype=dtype, requires_grad=True)
    
    gradcamtest = GradCAM(network=CNN, class_idx=5)
    print(gradcamtest.gradcam(img=a, class_idx=5))



  



    

    

    
    



    