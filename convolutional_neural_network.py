import numpy as np
import tensorflow as tf

class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

if __name__ == '__main__':
    # Example usage: Training a Convolutional Neural Network (CNN) on the MNIST dataset

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train)

    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_test = tf.keras.utils.to_categorical(y_test)

    # Build and train the CNN model
    input_shape = (28, 28, 1)
    num_classes = 10
    cnn = ConvolutionalNeuralNetwork(input_shape, num_classes)
    cnn.train(x_train, y_train, batch_size=128, epochs=5)

    # Evaluate the trained model
    accuracy = cnn.evaluate(x_test, y_test)[1]
    print("Accuracy:", accuracy)
