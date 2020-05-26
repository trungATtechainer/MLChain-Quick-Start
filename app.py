# import libraries
# pytorch dependencies


# support for image processing


# mlchain libraries



# redefine our model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 10, 5)
        # convolutional layer
        self.conv2 = nn.Conv2d(10, 20, 5)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (320-> 50)
        self.fc1 = nn.Linear(320, 50)
        # linear layer (50 -> 10)
        self.fc2 = nn.Linear(50, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        # flatten image input
        x = x.view(-1, 320)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

class Model():
    # define and load our prior model
    def __init__(self):

        # TODO: ADD CODE HERE

    # define function for predicting images
    def image_predict(self, img:np.ndarray):
        r"""
        Predict classes that image is based in
        Args:
            img(numpy array): Return an image used for prediction

        Note: You don't have to worry about the input for this function. Most of the time
        the input img:np.ndarray would be sufficient. It's important how you work with that input.
        """

       # TODO: ADD CODE HERE

# deploying our model
# define model
model = Model()

# serve model
serve_model = ServeModel(model)
