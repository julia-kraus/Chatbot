import tensorflow as tf
import tflearn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# reset underlying graph data

class TfModel:
    def __init__(self, train_x, train_y):
        # here, default learning rate is 0.001
        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
        self.model.save('model.tflearn')
        print("TensorFlow model is now trained")


class TorchModel(nn.Module):
    def __init__(self, train_x, train_y):
        super(TorchModel, self).__init__()
        self.fully_connected1 = nn.Linear(len(train_x[0]), 8)
        self.fully_connected2 = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.fully_connected1(x))
        x = self.fully_connected2(x)
        return F.log_softmax(x)

    def train(self, mode=True):
        # learning rate defaults to 1e-2
        optimizer = optim.SGD(self.parameters(), lr=0.001)
        criterion = nn.NLLLoss()
        print("PyTorch model is now trained")
