import torch
from torch.autograd import Variable

"""A fully connected network with two hidden layers"""

model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.Linear(),
          torch.nn.Softmax(H, D_out),
        )
    