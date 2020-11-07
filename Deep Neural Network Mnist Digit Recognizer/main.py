import numpy as np
import matplotlib.pyplot as plt


#------------------------------ LOAD DATA FUNCTION ------------------------------#
def load_data(train_data_path,  test_data_path):
    train_data = np.loadtxt(train_data_path, delimiter = ',') # An n-dimentional numpy array with shape (rows, columns) of Training Data
    test_data = np.loadtxt(test_data_path, delimiter = ',') # An n-dimentional numpy array with shape (rows, columns) of Testing Data

    train_data_img = train_data[:, 1:]
    train_data_labels = train_data[:, :1]
    test_data_img = test_data[:, 1:]
    test_data_labels = test_data[:, :1]

    return train_data_img.T, train_data_labels, test_data_img.T, test_data_labels


#------------------------------ WEIGHTS & BIASIS INTIALIZATION ------------------------------#
def initialize_parameters(layer_dims):
    L = len(layer_dims)
    W = {}
    b = {}

    for layer in range(1, L):
        # A Dictionary W which has 2D arrays representing Weights of each neuron in a Layer
        W[str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        # A Dictionary b which has 2D arrays representing Biasis of each neuron in a Layer
        b[str(layer)] = np.zeros((layer_dims[layer], 1))

    return W, b


#------------------------------ FORWARD PROPAGATION ------------------------------#
def layer_forward_propagation(A_prev, W, b, activation):
    if (activation.lower() == 'relu'):
        Z = np.dot(W, A_prev) + b
        A = np.maximum(0, Z)
    elif (activation.lower() == 'sigmoid'):
        Z = np.dot(W, A_prev) + b
        A = 1 / (1 + np.exp(-Z))

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    # A tupple of Activation from previous Layer, Weights and Biasis of current Layer and cache (Z) of current Layer
    cache = (A_prev, W, b, Z)

    return A, cache


def L_layer_forward_propagation(X, W, b):
    A = X
    caches = []
    L = len(W)

    for layer in range(1, L):
        A_prev = A
        A, cache = layer_forward_propagation(
            A_prev, W[str(layer)], b[str(layer)], "relu")
        caches.append(cache)

    AL, cacheL = layer_forward_propagation(A, W[str(L)], b[str(L)], "sigmoid")
    caches.append(cacheL)

    return AL, caches


#------------------------------ COMPUTING COST ------------------------------#
def compute_cost(AL, Y):
    m = len(Y)
    cost = - (1 / m) * np.sum(np.multiply(Y.T, np.log(AL)) + np.multiply((1 - Y.T), np.log(1 - AL)))
    cost = np.squeeze(cost)

    return cost


#------------------------------ BACKWARD PROPAGATION ------------------------------#
def layer_backward_propagation(dA, cache, activation):
    A_prev, W, b, Z = cache
    m = len(W)

    if (activation.lower() == "relu"):
        dZ = dA * np.where(Z <= 0, 0, 1)
    elif (activation.lower() == "sigmoid"):
        dZ = dA * (np.exp(Z) / np.power((np.exp(Z) + 1), 2))

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = (1 / m) * np.dot(W.T, dZ)

    return dA_prev, dW, db


def L_layer_backward_propagation(AL, Y, caches):
    dA = {}
    dW = {}
    db = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    dA[str(L - 1)], dW[str(L)], db[str(L)] = layer_backward_propagation(dAL, current_cache, "sigmoid")

    for layer in reversed(range(L - 1)):
        current_cache = caches[layer]
        dA_prev_temp, dW_temp, db_temp = layer_backward_propagation(
            dA[str(layer + 1)], current_cache, "relu")

        dA[str(layer)] = dA_prev_temp
        dW[str(layer + 1)] = dW_temp
        db[str(layer + 1)] = db_temp

    return dA, dW, db


#------------------------------ UPDATING WIEGHTS AND BIASIS ------------------------------#
def update_parameters(W, b, dW, db, learning_rate):
    L = len(W)

    for layer in range(1, L):
        W[str(layer)] = W[str(layer)] - np.multiply(learning_rate, dW[str(layer)])
        b[str(layer)] = b[str(layer)] - np.multiply(learning_rate, db[str(layer)])

    return W, b


#------------------------------ NEURAL NETWORK ------------------------------#
def L_layer_neural_network(X, Y, layer_dims, learning_rate, num_iterations):
    costs = []

    W, b = initialize_parameters(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_layer_forward_propagation(X, W, b)
        cost = compute_cost(AL, Y)
        dA, dW, db = L_layer_backward_propagation(AL, Y, caches)
        W, b = update_parameters(W, b, dW, db, learning_rate)

        if (i % 100 == 0):
            print("Cost after iteration %i is %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.title("Learning Rate = " + str(learning_rate))
    plt.show()

    return W, b


#------------------------------ DIGIT RECOGNIZER ------------------------------#
def digit_recognizer(X, W, b):
    Y, caches = L_layer_forward_propagation(X, W, b)
    return Y


#------------------------------ TRAINING CONTINUED ------------------------------#
def training_continued(W, b, X, Y, layer_dims, learning_rate, num_iterations):
    costs = []

    for i in range(num_iterations):
        AL, caches = L_layer_forward_propagation(X, W, b)
        cost = compute_cost(AL, Y)
        dA, dW, db = L_layer_backward_propagation(AL, Y, caches)
        W, b = update_parameters(W, b, dW, db, learning_rate)

        if (i % 100 == 0):
            print("Cost after iteration %i is %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.title("Learning Rate = " + str(learning_rate))
    plt.show()

    return W, b


#------------------------------ MAIN ------------------------------#
if __name__ == '__main__':
    trainingDataPath = "C:/Users/Maham/Documents/HS/Python Projects/Neural_Networks/Digit Recognizer/Training Data/mnist_train.csv"
    testingDataPath = "C:/Users/Maham/Documents/HS/Python Projects/Neural_Networks/Digit Recognizer/Testing Data/mnist_test.csv"

    trainDataImg, trainDataImgLabel, testDataImg, testDataImgLabel = load_data(trainingDataPath, testingDataPath)
    layerDims = [len(trainDataImg), 800, 10]

    trainDataLabel = (np.arange(10) == trainDataImgLabel).astype(np.float)
    trainDataLabel[trainDataLabel == 0] = 0
    trainDataLabel[trainDataLabel == 1] = 1

    learningRate = 0.0001
    numIterations = 5000

    #trainDataImgSeg = trainDataImg[:, 54000:]
    #trainDataLabelSeg = trainDataImgLabel[54000:, :]

    #img = trainDataImgSeg.T[571].reshape((28,28))
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    #print("Label: ", trainDataLabelSeg[571])

    #print(trainDataImgSeg.shape)
    #print(trainDataLabelSeg.shape)

    #trainDataLabelSeg = (np.arange(10) == trainDataLabelSeg).astype(np.float)
    #trainDataLabelSeg[trainDataLabelSeg == 0] = 0
    #trainDataLabelSeg[trainDataLabelSeg == 1] = 1

    W, b = L_layer_neural_network(trainDataImg, trainDataLabel, layerDims, learningRate, numIterations)


    #------------------------------ TRAINING CONTINUED ------------------------------#
    learningRate = 0.001
    numIterations = 5000

    trainDataImgSeg = trainDataImg[:, 54000:]
    trainDataLabelSeg = trainDataImgLabel[54000:, :]

    #img = trainDataImgSeg.T[571].reshape((28,28))
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    #print("Label: ", trainDataLabelSeg[571])

    #print(trainDataImgSeg.shape)
    #print(trainDataLabelSeg.shape)

    trainDataLabelSeg = (np.arange(10) == trainDataLabelSeg).astype(np.float)
    trainDataLabelSeg[trainDataLabelSeg == 0] = 0
    trainDataLabelSeg[trainDataLabelSeg == 1] = 1

    W, b = training_continued(W, b, trainDataImgSeg, trainDataLabelSeg, layerDims, learningRate, numIterations)

    
    #------------------------------ TRAINED MODEL ------------------------------#
    Y = digit_recognizer(trainDataImgSeg, W, b)

    for i in range(10):
        test = np.random.randint(1, 6001)
        print("Guesses: ", Y.T[test])
        print("Predicition: ", np.where(Y.T[test] == (Y.T[test].max()))[0][0])
        print("Actual Digit: ", testDataImgLabel[test])
        img = testDataImg.T[test].reshape((28, 28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        print("\n\n\n")
