# PyTorch official tutorial
https://pytorch.org/tutorials/beginner/basics/intro.html
The essential problem is that we do not know the coding format of pytorch, so going over pytorch tutorial and modifying it might be the best way to be masterful of it. 
##	Chapter 1, introduction to pytorch; 
###	1.0, LEARN THE BASICS; 
#### Running the Tutorial Code; 
#### How to Use this Guide; (2022-1-21)

###	1.1, QUICKSTART; 
#### Working with data; 
torch.utils.data.DataLoader and torch.utils.data.Dataset; (1) Dataset stores the samples and their corresponding labels, (2) DataLoader wraps an iterable around the Dataset; 
// the file format is py, not ipynb (short for ipython notebook); (2022-1-21)
// we can print the model’s properties; (2022-1-21)
// add del model according to leehongyi’s homework 1; (2022-1-21)
// use ############# to separate paragraph; (2022-1-21)

###	1.2, TENSORS; (2022-1-22)

###	1.3, DATASETS & DATALOADERS; 
Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples; (2022-1-22)

###	1.4, TRANSFORMS; 
ToTensor(); Lambda Transforms; (2022-1-22)

###	1.5, Build the neural network; 
gpu’s name is cuda; 1, Get Device for Training; 2, Define the Class; 3, Model Layers; 4, Model parameters; (2022-1-23)
// replace nn.Sequential with one layer and one layer; (2022-1-24)

###	1.6, AUTOGRAD; 
1, Tensors, Functions and Computational graph; 2, Computing Gradients; 3, Disabling Gradient Tracking; (2022-1-23)

###	1.7, Optimization; 
1, Prerequisite Code; 2, Hyperparameters; 4, Full Implementation; (2022-1-23)
// replace loss_fn with criterion; replace (X, y) with data; replace batch with i; (2022-1-24)
// replace nn.Sequential with one layer and one layer; (2022-1-24)

###	1.8, save & load; 
1. Saving and Loading Model Weights; 
2. Saving and Loading Models with Shapes; (2022-1-23)
	

##	Chapter 2, Introduction to PyTorch on YouTube
###	2.0, INTRODUCTION TO PYTORCH - YOUTUBE SERIES; (2022-1-23)

###	2.1, INTRODUCTION TO PYTORCH; 
1. increase epoch to 10 to check whether result improves; 
2. tensor’s definition is according to dimension, not in mathematics; (2022-1-24)
3. dtype means data type; (2022-1-24)
4. assert condition == if not condition: raise AssertionError() (2022-1-25)

###	2.2, THE FUNDAMENTALS OF AUTOGRAD; 
	
	

