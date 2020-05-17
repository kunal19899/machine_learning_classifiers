'''
Kunal Samant
1001534662
'''

import sys
import os
import numpy as np
import math

def normalize(data, mean, std):
    new_data = []
    for x in data:
        new_line = []
        for i in range(len(x)-1):
            new_line.append((x[i]-mean[i])/std[i])
        new_line.append(x[-1])
        new_data.append(new_line)
    new_data = np.array(new_data)
    return new_data

def find_mean_and_std(data, D):
    mean = []
    std = []
    for i in range(D):
        x = data[:, i]
        mean.append(np.mean(x))
        std.append(np.std(x))
    return mean, std

def euclidian(object, training_data):
    object_without_class = object[:len(object)-1]
    training_data_without_class = training_data[:, :len(training_data[0])-1]
    object_without_class = [object_without_class]*len(training_data_without_class)
    neighbors = np.sqrt(np.sum((object_without_class - training_data_without_class)**2, axis=1))
    neighbors = list(zip(neighbors, training_data[:, -1]))
    return neighbors

def knn_classify(training_data, test_data, k):
    training = np.loadtxt(training_data)
    D = len(training[0]-1)
    mean, std = find_mean_and_std(training, D)
    new_data = normalize(training, mean, std)
    
    test = np.loadtxt(test_data)
    new_test_data = normalize(test, mean, std)

    accuracy = []
    prediction = []
    count = 0
    
    for x in new_test_data:
        neighbors = euclidian(x, new_data)
        neighbors = sorted(neighbors, key=lambda neighbors: neighbors[0])
        neighbors = neighbors[:k]
        output_values = [row[-1] for row in neighbors]
        predict = max(set(output_values), key=output_values.count)
        prediction.append(predict)
        if predict == x[-1]:
            accuracy.append(1)
            count += 1
        else:
            accuracy.append(0)

    for i in range(len(test)):
        print("ID=" + "%5d"%int(i+1) + ", predicted=" + "%3d"%prediction[i] + ", true=" + "%3d"%test[i][-1] + ", accuracy=" + "%4.2f"%accuracy[i])
    
    print("classification accuracy=" + "%6.4f"%(count / len(test)))


training_data = sys.argv[1]
test_data = sys.argv[2]
k = sys.argv[3]

knn_classify(training_data, test_data, int(k))