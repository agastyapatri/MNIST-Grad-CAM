# **Implementing Grad-CAM on a MNIST Classification Model**

_This project aims at implementing **Grad-CAM** on a simple Convolutional Neural Network_

## **1. About the Project**

* The MNIST databse [1] of handwritten digits has a training set of 60,000 examples. Each sample is a 28x28 black-and-white image, and one class per digit.  

While a classification model featuring a (relatively simple) Convolutional Neural Network will be built, the main purpose is to implement **Grad-CAM** for self-edification

### **Grad-CAM, or _Gradient-Weighted Class Activation Mapping_.**
Grad-CAM is a a class discriminative visualization technique [2], meaning that it produces a visualization which highlights the regions of the input image that are important for its classification into a certain label.  

For Example, given that the class of the image is "Dog", Grad-CAM is able to identify the parts of the image which contribute most to the identification of the image as that of a "Dog". 

## **2. Modus Operandi**
Crudely:
    
1. Create a ConvNet to classify MNIST.
2. Measure how it performs.
3. Implement Grad-CAM.
4. Use Grad-CAM to see where the model is misclassifying the images.
5. Improve Model 
6. Report Results.

_Re-Write this section_

### **2.1: Creating the ConvNet**
While MNIST classification is the very first thing everyone cuts their teeth on, it is still useful to revisit the dataset with more sophistication, both as a programmer and as a practitioner. 

Structure of the ConvNet built: 
    
  1. Conv2d(1, 12, kernel_size=(5, 5), stride=(1, 1), padding=(None, None))
  2. ReLU()
  3. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  4. Flatten(start_dim=1, end_dim=-1)
  5. Linear(in_features=128, out_features=10, bias=True)
  6. Sigmoid()
  7. Linear(in_features=1, out_features=2, bias=True)
  8. Softmax(dim=None)


_Note that this is temporary, and the parameters are not final (or even correct)_




### **2.2 Grad-CAM all the way**







## **3. Results and Discussion**


## **4. Notes**

_Currently, a list of issues faced:_

1. PyTorch's Conv2D class doesn't seem to be implemented for the datatype torch.float16. This datatype was chosen to alleviate performance burden at runtime, but float32 it is. 
Turns out that float16 would have been slower anyway due to software conversion to float32, as I dont have tensor cores at hand. Whatever that means.

2.   



-----------------------------------
## **References**
[1] : [The MNIST Dataset, from Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)

[2]: Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2020). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. _International Journal of Computer Vision_, [online] 128(2), pp.336–359. doi:10.1007/s11263-019-01228-7.

‌[3] : [The Grad-CAM Website](http://gradcam.cloudcv.org/)