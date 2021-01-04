##------------------------------ IMPORTS ------------------------------##
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


##------------------------------ TRAIN-TEST DATA ------------------------------##
(trainDataImg, trainDataLabel), (testDataImg, testDataLabel) = mnist.load_data()

trainDataImg = trainDataImg.reshape(-1, 784)                        # Reshaping the Train Images into Automatical adjusted dimenstion (-1) and 784 Units
testDataImg = testDataImg.reshape(-1, 784)                          # Reshaping the Test Images into Automatical adjusted dimenstion (-1) and 784 Units

trainDataLabel = to_categorical(trainDataLabel, num_classes=10)     # to_categorical() Function takes in Targets and return one-hot encoding off that
testDataLabel = to_categorical(testDataLabel, num_classes=10)       # to_categorical() Function takes in Targets and return one-hot encoding off that


##------------------------------ FEED-FORWARD MODEL ------------------------------##
digitRecognizer = tf.keras.models.Sequential(                                                       # Using Sequential Model from Keras API of Tensorflow
    [
        tf.keras.layers.Dense(2500, activation="relu", input_shape=(784,)),                         # Creating Input Layer with Density of 2500 units/neurons, relu activation and input size of 784
        tf.keras.layers.Dense(2000, activation="relu"),                                             # Creating 1st Hidden Layer with Density of 2000 hidden units/neurons and relu activation
        tf.keras.layers.Dense(1500, activation="relu"),                                             # Creating 2nd Hidden Layer with Density of 1500 hidden units/neurons and relu activation
        tf.keras.layers.Dense(1000, activation="relu"),                                             # Creating 3rd Hiddel Layer with Density of 1000 hidden units/neurons and relu activation
        tf.keras.layers.Dense(500, activation="relu"),                                              # Creating 4th Hidden Layer with Density of 500 hidden units/neurons and relu activation
        tf.keras.layers.Dense(10, activation="softmax")                                             # Creating Output Layer with Density of 10 units/neurons and softmax activation
    ]
)

digitRecognizer.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])    # Compiling the Neural Network Model we Created by defining the loss-type (Categorical Cross Entropy), Optimizer (adma) and performance analyser funtion based on a property (accuracy)
digitRecognizer.summary()                                                                           # Summary of the Model


##------------------------------ TRAINING ------------------------------##
digitRecognizerHistory = digitRecognizer.fit(trainDataImg, trainDataLabel, batch_size=1200, epochs=5)   # Fiting our Model to the Training Data and Training Labels. fit() function takes in batch_size and epochs to train the model


##------------------------------ PREDICTION ------------------------------##
if __name__ == '__main__':
    prediction = digitRecognizer.predict(testDataImg)

    for i in range(10):
        testImgNumber = np.random.randint(0, 10001)

        print("Image Number: ", testImgNumber)
        print("Predictions Weightage: ", prediction[testImgNumber])
        print("Actual Weightage: ", testDataLabel[testImgNumber])
        print("Predicted Number: ", np.where(prediction[testImgNumber] == prediction[testImgNumber].max())[0][0])   # Printing the Largest Number in the Predictions at an index
        print("Actual Digit:", np.where(testDataLabel[testImgNumber] == testDataLabel[testImgNumber].max()[0][0]))  # Printing the Largest Number in the Test Data Labels at an index

        digitImg = testDataImg[testImgNumber].reshape((28, 28))
        plt.imshow(digitImg, cmap="Greys")
        plt.show()
        print("\n\n\n")