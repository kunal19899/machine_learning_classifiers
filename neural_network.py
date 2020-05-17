import sys
import os
import numpy as np
import random


def sigmoid(x):
    return 1/(1+np.exp(-x))

def perceptron(weights, x):
    return sigmoid(np.dot(np.transpose(weights),x))
    
def createWeights(rest):
    weights = []
    for i in range(rest):
        weights.append(random.uniform(-0.05, 0.05))
    return weights

def updatePerceptronAndCreateWeights(start, end, z, units, nums, create):
    update = []
    updateTotWeights = []
    for i in range(units):
        weights = createWeights(nums)
        up = [random.uniform(-0.05, 0.05)]
        updateWeights = up + weights
        update.append(perceptron(updateWeights, z[start: end]))
        updateTotWeights.append(updateWeights)
    return update, updateTotWeights

def updatePerceptron(units, nodes, layer, nums, weights, z, start, end):
    update = []
    for i in range(nodes):
        update.append(perceptron(weights[layer*units + i], z[start: end]))
    return update

def make_t(number, classes):
    t = []
    for i in range(classes):
        if i == number:
            t.append(1)
        else:
            t.append(0)
    return t

def updateSigma(z, t, numPerceptrons, start, end):
    sigma = []
    for j in range(start, end, -1):
        val = (z[j]-t[start-end-1]) * z[j] * (1-z[j])
        sigma.append(val)
    return sigma[::-1]

def updateSigma_hidden(U, D, z, weights, sigma, start, end, urange):
    val = []
    for j in range(start, end, -1):
        sums = 0
        for u in range(urange):
            sums += (sigma[U - u - 1] * weights[(U-D-1)][j-end-1])
        val.append(sums * z[j] * (1-z[j]))
    return val[::-1]

def recalibrate(weights, sigma, z, nums, prenums, n, l, D, units):
    newWeights = []
    for i in range(nums):
        indiWeight = []
        for j in range(prenums):
            indiWeight.append((weights[i][j] - (n * sigma[D + (units*l) + i] * z[D + (units*l) + i])))
        newWeights.append(indiWeight)
    return newWeights

def neural_network(training_data, test_data, layers, units, rounds):
    if int(layers) < 2:
        print("Error: Layers cannot be less than 2!")
    layers = layers-2
    training_data = np.loadtxt(training_data)
    create = True
    bias_input, learning_rate = 1, 1
    D = len(training_data[0])
    training_data_without_class = []
    t = []
    classes = []

    # normalize dataset
    new_data = []
    for loop in range(2):
        if loop == 0:
            high = 0
            for x in training_data:
                x = x[:len(x)-1]
                if high < np.max(x):
                    high = np.max(x)
        if loop == 1:
            for i in range(len(training_data)):
                if training_data[i][len(training_data[i])-1] not in classes:
                    classes.append(training_data[i][len(training_data[i])-1])
                new_data.append(np.insert(training_data[i], 0, bias_input, axis=0))
                new_data[i] = new_data[i]/high
                training_data_without_class.append(new_data[i][:len(new_data[i])-1])
    
    for x in training_data:
        t.append(make_t(x[D-1], len(classes)))

    U = D + int(layers)*int(units) + len(classes)

    numWeights = (D*units) + (np.square(units) * (layers-1)) + (len(classes)*units) + (layers+1)

    weights = [np.zeros(numWeights)]

    for round in range(rounds):
        iterator = 0
        for x in training_data_without_class:  
            z = np.zeros(U)
            # print(len(weights))
            for j in range(D):
                z[j] = x[j]

            for i in range(len(weights)):
                if weights[0][i] != np.zeros(numWeights)[i]:
                    create = False
                    break

            # print(len(weights))
            for l in range(layers+1):
                if l == 0:
                    if create is True:
                        z[D:D+units], updateWeights = updatePerceptronAndCreateWeights(0, D, z, units, D-1, create)
                        weights[:(D*units)+1] = updateWeights
                    else:
                        z[D:D+units]= updatePerceptron(units, units, l, D, weights, z, 0, D)
                elif l == layers:
                    if create is True:
                        #U-len(classes)-units+1, U-len(classes)+1
                        with_bias = np.ones(units+1)
                        with_bias[1:] = z[U-len(classes)-units+1: U-len(classes)+1]
                        z[U-len(classes): U+1], updateWeights = updatePerceptronAndCreateWeights(0, len(classes) + 1, with_bias, len(classes), units, create)
                        weights[numWeights - units*len(classes)-1:] = updateWeights
                    else:
                        with_bias = np.ones(units+1)
                        with_bias[1:] = z[U-len(classes)-units+1: U-len(classes)+1]
                        z[U-len(classes): U+1] = updatePerceptron(units, len(classes), l, units, weights, with_bias, 0, len(classes))
                else:
                    if create is True:
                        # print(z[D + units*(l-1): (D + units*(l-1)+units)])
                        with_bias = np.ones(units+1)
                        with_bias[1:] = z[D + units*(l-1): (D + units*(l-1)+units)]
                        z[D + (units*(l)) : D + (units*(l)) + units], updateWeights = updatePerceptronAndCreateWeights(0, units+1, with_bias, units, units, create) 
                        weights[(D*units) + np.square(units) * (layers) + 1: (D*units) + np.square(units) * (layers) + np.square(units)] = updateWeights
                    else:
                        with_bias = np.ones(units+1)
                        with_bias[1:] = z[D + units*(l-1): (D + units*(l-1)+units)]
                        z[D + (units*(l)) : D + (units*(l)) + units] = updatePerceptron(units, units, l, units, weights, with_bias, 0, units+1)
            sigma = np.zeros(U)
            r = U - len(classes)

            for l in range(layers+1, 0, -1):
                if l == layers+1:
                    sigma[r:] = updateSigma(z, t[iterator], len(classes), U-1, r-1)
                elif l+1 == layers+1:
                    sigma[r-units: r] = updateSigma_hidden(U, D, z, weights, sigma, (D+(units*l)-1), (D+(units*l)-1)-units, len(classes))
                else:
                    sigma[r - (units * (layers+1 - l)): r - (units * (layers+1 - l))+units] = updateSigma_hidden(U, D, z, weights, sigma, (D+(units*l)-1), (D+(units*l)-1)-units, units)


            for l in range(layers+1):
                if l == 0:
                    weights[:units] = recalibrate(weights, sigma, z, units, D, learning_rate, l, D, units)
                
                elif l == layers:
                    weights[D-len(classes)+1:] = recalibrate(weights, sigma, z, len(classes), units+1, learning_rate, l, D, units)
                else:
                    weights[units*l : units*l + units] = recalibrate(weights, sigma, z, units, units+1, learning_rate, l, D, units)
        

            iterator += 1
            learning_rate *= 0.98


    test_data = np.loadtxt(test_data)
    test_data_without_class = []
    test_t = []
    test_classes = []
    test_z = np.zeros(U)
    
    for loop in range(2):
        if loop == 0:
            high = 0
            for x in test_data:
                x = x[:len(x)-1]
                if high < np.max(x):
                    high = np.max(x)
    if loop == 1:
        for i in range(len(test_data)):
            if test_data[i][len(test_data[i])-1] not in classes:
                test_classes.append(test_data[i][len(test_data[i])-1])
            new_data.append(np.insert(test_data[i], 0, bias_input, axis=0))
            new_data[i] = new_data[i]/high
            test_data_without_class.append(new_data[i][:len(new_data[i])-1])

    for x in test_data:
        test_t.append(make_t(x[D-1], len(classes)))

    for x in test_data_without_class:
        for j in range(D):
            test_z[j] = x[j]

        for l in range(layers+1):
            if l == 0:
                test_z[D:D+units] = updatePerceptron(units, units, l, 0, weights, z, 0, D) 
            elif l == layers:
                with_bias = np.ones(units+1)
                with_bias[1:] = z[U-len(classes)-units+1: U-len(classes)+1]
                test_z[U-len(classes): U+1] = updatePerceptron(units, len(classes), l, units, weights, with_bias, 0, len(classes))
            else:
                with_bias = np.ones(units+1)
                with_bias[1:] = z[D + units*(l-1): (D + units*(l-1)+units)]
                test_z[D + (units*(l)) : D + (units*(l)) + units] = updatePerceptron(units, units, l, units, weights, with_bias, 0, units+1)

        print(test_z)


        


    




training_data = sys.argv[1]
test_data = sys.argv[2]
layers = sys.argv[3]
units = sys.argv[4]
rounds = sys.argv[5]


neural_network(training_data, test_data, int(layers), int(units), int(rounds))