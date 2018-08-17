import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducible
seed = 7
numpy.random.seed(seed)

#load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28 x 28 images to a 784 vector for each image
# and cast them to pixel value precision of 32 to reduce memory requirements
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
x_test = X_test / 255

# one hot encode outputs to transform the vector of class integers to a binary matrix
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the baseline model
def baseline_model():
    # create model: simple neural network with one hidden layer with the same number of neurons as there are inputs (784)
    # a rectifer activation function is used for the neurons in the hidden layer
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal'))
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # softmax activation function on output layer to turn outputs in probability values and allow one class of 10 to be selected
    # as the model's output prediction
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model - Logarithmic loss as loss function and ADAM gradient descent algorithm to learn weights
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline error: %.2f%%" % (100-scores[1]*100))



