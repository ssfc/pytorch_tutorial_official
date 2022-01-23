# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

import torch
import torchvision.models as models

###########################################################################################################################################################
# 1: Saving and Loading Model Weights
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')  # PyTorch models store the learned parameters in an internal state dictionary, called state_dict. 

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))  # To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.
model.eval()

###########################################################################################################################################################
# 2: Saving and Loading Models with Shapes


































