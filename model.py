"""
CNN model for MNIST classification for CS420 Data Mining
Base code provided by Huajie Shao, Ph.D
Model and feedforward defined by David Ni
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
	def __init__(self,input_channel, out_channel,my_kernel, my_stride, dropout=0.1):
		super(CNNModel, self).__init__()
		## CNN layers
		self.conv = nn.Sequential(nn.Conv2d(input_channel, out_channel, my_kernel, my_stride),
									#batch parameter = output channel size
									nn.BatchNorm2d(8),
									nn.ReLU(),
									nn.Dropout(dropout), 
									# second layer
									nn.Conv2d(8, 15, 2, 2),
									nn.BatchNorm2d(15),
									nn.ReLU(),
									nn.Dropout(dropout), 
									# third layer
									nn.Conv2d(15, 20, 2, 1),
									nn.BatchNorm2d(20),
									nn.ReLU(),
									nn.Dropout(dropout),

									# maxpool
									nn.MaxPool2d(2,1),
									nn.ReLU()
									)
		in_size = 20*5*5  #output of Conv2d
		hidden_size = 40 # size of output of hidden layer
		out_size = 10 # size of final

		# first hidden layer
		self.fc = nn.Linear(in_size, hidden_size)
		# second fully connected/output layer
		# use two hidden layers to improve learning
		self.fc2 = nn.Linear(hidden_size, out_size)
		

	# Feedforward function
	def forward(self, x): 
		# preform convolutions on input
		x_out = self.conv(x)
		## write flatten convultions
		flat = torch.flatten(x_out,1) # x_out is output of last layer

		# Pass convolution output to hidden layers
		# hidden layer
		hidden_layer =  self.fc(flat) # predict y
		# output layer
		result = self.fc2(hidden_layer) # predict y
		
		return result
        
		
		
	
		