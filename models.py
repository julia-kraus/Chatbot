import tensorflow as tf
import tflearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
        self.model.save('model.tflearn')
        print("TensorFlow model is now trained.")


class TorchModel(nn.Module):
    def __init__(self, train_x, train_y):
        super(TorchModel, self).__init__()
        self.fully_connected1 = nn.Linear(len(train_x[0]), 8)
        self.fully_connected2 = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.fully_connected1(x))
        x = self.fully_connected2(x)
        return F.log_softmax(x)

    def fit(self, x_train, y_train):
        # learning rate defaults to 1e-2
        # choose optimizer and loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        num_epochs = 1000

        # train
        for epoch in range(num_epochs):
            X = Variable(torch.Tensor(x_train).float())
            Y = Variable(torch.Tensor(y_train).long())

            # feedforward = backprop
            # zero_grad clears the gradients of all optimized Variables
            optimizer.zero_grad()
            out = self(X)
            loss = criterion(out, torch.max(Y, 1)[1])
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print('Epoch [%d/%d] Loss: %.5f'
                      % (epoch + 1, num_epochs, loss.data[0]))
        print("Pytorch model is now trained.")
        torch.save(self.state_dict(), 'model.pytorch')
