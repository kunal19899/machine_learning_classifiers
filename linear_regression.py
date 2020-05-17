# Kunal Samant
# 1001534662

import os
import sys
import numpy as np
import pandas as pd
import sklearn as skl
import math

def exitFunction(statement):
    print("Error: " + statement)
    exit(0)

def training(training_input, degree, lambdaVal):
    training_input_without_class = [] # contains all the attributes of an object without its class 
    class_list = [] # contains all classes of training data
    for i in range(len(training_input)):
        training_input_without_class.append(training_input[i][:len(training_input[0])-1])
        class_list.append(training_input[i][len(training_input[i])-1])
    
    # calculate number pf attr in phibased on degree
    numRows = len(training_input_without_class[0])*degree + 1
    
    # initialize the phi table with ones
    phi = np.ones((len(training_input_without_class), numRows))
    
    # update phi table
    for i in range(len(training_input_without_class)): 
        j = 1
        attr = 0
        while j < numRows:
            for k in range(1, degree+1):
                phi[i][j] = math.pow(training_input_without_class[i][attr], k)
                j+=1
            attr += 1 

    #tranpose phi table
    phi_T = np.transpose(phi)

    # find the dot product of phi_T and phi
    w = np.dot(phi_T, phi)

    # identity function
    identity = np.identity(len(w))

    # finding the list of all weights
    minimised_w = np.dot(np.dot(np.linalg.pinv(lambdaVal*identity + w), phi_T), class_list)

    return minimised_w
    
def test(test_input, minimized_w, degree):
    test_input_without_class = [] # contains all the attributes of an object without its class 
    class_list = [] # contains all classes of training data
    expected_results = [] # list of the predicted results

    for i in range(len(test_input)):
        test_input_without_class.append(test_input[i][:len(test_input[0])-1])

    # calculate number pf attr in phibased on degree
    numRows = len(test_input_without_class[0])*degree + 1
    
    # initialize the phi table with ones
    phi = np.ones((len(test_input_without_class), numRows))
    
    # update phi table
    for i in range(len(test_input_without_class)): 
        j = 1
        attr = 0
        while j < numRows:
            for k in range(1, degree+1):
                phi[i][j] = math.pow(test_input_without_class[i][attr], k)
                j+=1
            attr += 1 

    squared_errors = []
    for i in range(len(test_input_without_class)):
        expected_results.append(np.dot(minimized_w, phi[i]))
        squared_errors.append(np.square(test_input[i][len(test_input[i])-1] - expected_results[i]))

    return expected_results, squared_errors

    


def linear_regression(training_data, degree, lambdaVal, test_data):
    if (int(degree) > 9 or int(degree) < 1):
        exitFunction("Degree needs to be between 1 and 10.")

    if (int(lambdaVal) < 0):
        exitFunction("Lambda must be a non-negative number")

    training_input = np.loadtxt(training_data)
    minimized_w = training(training_input, int(degree), int(lambdaVal))

    # print training results
    for i in range(len(minimized_w)):
        print("w" + str(i+1) + "=" + "%0.4f"%minimized_w[i])

    test_input = np.loadtxt(test_data)
    results, errors = test(test_input, minimized_w, int(degree))

    # print test results
    for i in range(len(test_input)):
        print("ID=" + "%5d"%int(i+1) + ", output=" + "%14.4f"%results[i] + ", target value=" + "%10.4f"%test_input[i][len(test_input[i])-1] + ", squared error=" + "%0.4f"%errors[i])



training_data = sys.argv[1]
degree = sys.argv[2]
lambdaVal = sys.argv[3]
test_data = sys.argv[4]

linear_regression(training_data, degree, lambdaVal, test_data)
