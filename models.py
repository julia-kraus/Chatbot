import tensorflow as tf
import tflearn


# reset underlying graph data

class TfModel:
    def __init__(self, train_x, train_y):
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
        print("Tensorflow model is now trained")

# class TorchModel(nn.Module):
#     def __init__(self):
#         super(TorchModel, self).__init__()
#         self.fully_connected1 = nn.Linear()
#         self.fully_connected2 = nn.Linear()
#         self.fully_connected3 = nn.Softmax()
#
#     def forward(self, *input):
#         pass
#
#     def back
