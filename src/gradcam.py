import numpy as np 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt  
from PIL import Image
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

    def __init__(self, network):

        super().__init__()
        self.net = network 
        self.classes = range(10)
        self.gradients = None 

        # breaking down the network into usable chunks. 
        self.CONV = self.net[0]

        self.FC = self.net[1] 





    def L_GC(self, image_data, class_name):
        image_batch, label_batch = image_data

        # features: (64 x 24 x 4 x 4)
        features = self.CONV(image_batch)

        # batches, num_feature_maps, height_of_feature_map, width_of_feature_map
        _, N, H, W = features.size()

        # outputs : (64x10). choose the output for the desired class. 
        outputs = self.FC(features)

        
        # finding the outputs, features for the desired image index
        for i in range(len(label_batch)):
            if class_name == label_batch[i]:

                # outputs(10) , features(24x4x4), image_tensor(1,28,28)
                image_tensor = image_batch[i]
                outputs_for_class = outputs[i]
                features_for_class = features[i]
         
        c_score = outputs_for_class[class_name]

        # gradients of output logits with respect to last feature maps 
        grads = torch.autograd.grad(c_score, features)[0][0]

        # finding neuron importance weights by global average pooling
        alphas = grads.mean(dim=-1).mean(dim=-1)
        features_for_class = features_for_class.view(features_for_class.size(0), -1)

        # Grad-CAM Scores by dot_product + applying relu
        L_c_intermediate = torch.matmul(alphas, features_for_class).detach().numpy()
        L_c_intermediate = L_c_intermediate.reshape(H, W) 
        L_c = np.maximum(L_c_intermediate, 0)

        return L_c

"""
    def plotGCAM(self, gcscores):
        # image = image_batch[0][0].detach().numpy()
        # image is the [28x28] tensor 
        L_c = gcscores
        sal = Image.fromarray(L_c)
        sal.resize(image.shape, resample=Image.LINEAR)

        plt.title("test")
        plt.axis("off")
        # plt.imshow(image)
        plt.imshow(np.array(sal), alpha=0.5, cmap="jet")
        plt.show()
        
"""        

        
        

    






if __name__ == "__main__":


    CONV = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        
            # Conv Layer 2 
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, dtype=dtype),
            nn.ReLU()

        ) 

    FCLayers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
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




    a = torch.randn(64, 1, 28, 28, dtype=dtype, requires_grad=True)
    b = torch.ones(1, 28, 28, dtype=dtype, requires_grad=True)

    a_labels = torch.ones(64)
    

    # testing for one image tensor of size (1,28,28)
    gradcamtest = GradCAM(network=CNN)
    # print(gradcamtest.L_GC(image_data=(a, a_labels), class_name=1))






  



    

    

    
    



    