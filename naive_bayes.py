# Kunal Samant
# 1001534662


import os
import sys
import numpy as np
import pandas as pd


# Calcultes mu and sigma for each attribute of each class
############################################   
def mu_and_sigma(data):
    df = pd.DataFrame(data)
    df.rename(columns = {len(data[0])-1: "class"}, inplace=True)
    averages = df.groupby(['class']).mean()
    std = df.groupby(['class']).std()
    groups = df.groupby(['class']).groups

    return averages, std, df, groups


####### Calculates Gaussian Value ##########

def gaussian(x, mu=0.0, sigma=1.0):
    x = float(x - mu) / sigma
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / sigma

###############################################

# p(C)
def p_Class(data):
    class_set = {}
    class_probability = {}
    for i in range(len(data)):
        if data[i][len(data[i])-1] not in class_set:
            class_set[data[i][len(data[i])-1]] = 0
        else:
            class_set[data[i][len(data[i])-1]] += 1
    
    for i in class_set:
        class_probability[i] = class_set[i] / len(data)

    return class_probability

# Data Training
###############################################
def train(training_data):
    training = np.loadtxt(training_data)

    ######### training the mean and standard deviation #######
    averages, stds, df, groups = mu_and_sigma(training)
    numAttr = 0
    while True:
        try:
            currAttr = averages[numAttr]
            numAttr += 1
        except:
            break

    ####### prints the training output ########
    print("\nTRAINING OUTPUT\n")
    for i in groups.keys():
        for j in range(numAttr):
            if stds[j][i] < 0.01:
                stds[j][i] = 0.01
            print("Class " + str(i) + ", attribute " + str(j+1) + ", mean = " + str("%.2f"%averages[j][i]) + ", std = " + str("%.2f"%stds[j][i]))
    
    class_probability = p_Class(training)

    return averages, stds, df, class_probability

###############################################


# Data Testing
###############################################
def test(test_data, averages, stds, df, class_probability):
    test = np.loadtxt(test_data)
    test_attr = []
    for data_point in test:
        test_attr.append(data_point[:len(test[0])-1])
    
    predicted_values = []
    probabilty_current = []

    sorted_classes = sorted(class_probability)
    for i in range(len(test_attr)):
        gaussians = []
        for j in range(len(sorted_classes)):
            # print(2)
            gaussians.append(1)
            for k in range(len(test_attr[i])):
                gaussians[j] *= gaussian(test_attr[i][k], averages[k][sorted_classes[j]], stds[k][sorted_classes[j]])
            gaussians[j] = gaussians[j]*class_probability[sorted_classes[j]]

        # use of sum rule to calculate P(x) and hence calculate the P(x|C) for each class
        gaussians[:] = [x/np.sum(gaussians) for x in gaussians]

        # maxi stores the max_values (more than one element if a tie exists)
        maxi = []
        for i in range(len(gaussians)):
            if len(maxi) == 0:
                maxi.append(gaussians[i])
            elif gaussians[i] == maxi[0]:
                maxi.append(gaussians[i])
            elif gaussians[i] > maxi[0]:
                maxi = []
                maxi.append(gaussians[i])

        # finds the index of probabilty AKA class
        curr_set = {}  
        for prob in maxi:
            curr_set[gaussians.index(prob)+1] = prob
        predicted_values.append(curr_set)
    
    # list of accuracies of each object
    accuracy = []
    print("\nTEST OUTPUT\n")
    for i in range(len(test)):
        predicted = 0
        key = list(predicted_values[i].keys())
        if (test[i][len(test[i])-1] in key):
            predicted = int(test[i][len(test[i])-1])
            accuracy.append(1/len(key))
        else:
            predicted = int(key[0])
            accuracy.append(0)
        print("ID = " + str("%5d"%(i+1)) + ", predicted = " + str("%3d"%predicted) + ", probabilty = " + str("%.4f"%predicted_values[i][predicted]) + ", true = " + str("%3d"%test[i][len(test[i])-1]) + ", accuracy = " + str("%4.2f"%accuracy[i]))
    classification_accuracy = np.mean(accuracy)*100
    print("\nClassification Accuracy = " + "%6.4f"%classification_accuracy + "%")


def naive_bayes(training_file, test_file):
    averages, stds, df, class_probability = train(training_file)
    test(test_file, averages, stds, df, class_probability)

training_file = "./yeast_training.txt"
test_file = "./yeast_test.txt"

naive_bayes(training_file, test_file)
