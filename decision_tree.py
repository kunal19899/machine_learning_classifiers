import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy


class Node:

    def __init__(self, attribute, threshold):
        self.attribute = attribute
        self.threshold = threshold
        self.gain = None
        self.distribution = None
        self.left_child = None
        self.right_child = None


def check_option(option):
    if option not in ['optimized', 'randomized', 'forest3', 'forest15']:
        sys.exit(0)

def DISTRIBUTION(examples, classes):
    num_classes = int(np.amax(classes) - np.amin(classes) + 1)
    count = np.zeros(num_classes)
    x = examples[:, -1]

    for i in range(len(x)):
        count[int(x[i])] += 1

    return np.array(count)/len(examples)

def SELECT_COLUMN(examples, attribute):
    return np.array(examples[:, attribute])

def INFORMATION_GAIN(examples, A, threshold, classes): #have to implement this
    initial_entropy = 0
    entropy_right = 0
    entropy_left = 0
    
    initial = DISTRIBUTION(examples, classes)
    for i in range(len(initial)):
        if initial[i] != 0:
            initial_entropy -= initial[i]*math.log2(initial[i])

    check_list = examples[:, A]

    left = []
    right = []

    for i in range(0, len(check_list), 1):
        if check_list[i] < threshold:
            left.append(examples[i])
        else:
            right.append(examples[i])

    if len(left) == 0:
        entropy_left = 0
    else:
        entropy_left = entropy(DISTRIBUTION(np.array(left), classes), base=2)

    if len(right) == 0:
        entropy_right = 0
    else:
        entropy_right = entropy(DISTRIBUTION(np.array(right), classes), base=2)
    
    return initial_entropy - (len(left)/len(examples))*entropy_left - (len(right)/len(examples))*entropy_right    

def randomized(examples, attributes, classes, pruning_thr):
        max_gain = -1
        best_threshold = -1
        A = np.random.randint(0, max(attributes))
        attribute_values = SELECT_COLUMN(examples, A)
        L = min(attribute_values)
        M = max(attribute_values)

        for K in range(1, 51):
            threshold = (L+(K*(M-L))) / 51
            gain = INFORMATION_GAIN(examples, A, threshold, classes)
            if gain > max_gain:
                max_gain = gain
                best_threshold = threshold

        return A, best_threshold, max_gain

def CHOOSE_ATTRIBUTE(examples, attributes, classes, pruning_thr):
    if option == 'optimized':
        max_gain = -1
        best_attribute = -1
        best_threshold = -1
        for A in attributes:
            attribute_values = SELECT_COLUMN(examples, A)
            L = min(attribute_values)
            M = max(attribute_values)

            for K in range(1, 51):
                # print(K)
                threshold = (L+(K*(M-L))) / 51
                gain = INFORMATION_GAIN(examples, A, threshold, classes)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = A
                    best_threshold = threshold

        return best_attribute, best_threshold, max_gain
    
    elif option == 'randomized' or option == 'forest3' or option == 'forest15':
        return randomized(examples, attributes, classes, pruning_thr)


def allSame(examples):
    for i in range(1, len(examples)):
        if examples[i][-1] != examples[i-1][-1]:
            return False
    return True

def DTL(examples, attributes, default, pruning_thr, classes):
    if len(examples) < pruning_thr:
        node = Node(-1, -1)
        node.distribution = default
        node.gain = 0
        return node

    elif allSame(examples):
        node = Node(-1,-1)
        final = np.zeros(int(max(classes)-min(classes)+1))
        final[int(examples[0][-1] - min(classes))] = 1
        node.distribution = final
        node.gain = 0
        return node

    else:
        best_attribute, best_threshold, max_gain = CHOOSE_ATTRIBUTE(examples, attributes, classes, pruning_thr)
        tree = Node(best_attribute, best_threshold)
        tree.gain = max_gain
        tree.distribution = DISTRIBUTION(examples, classes)

        examples_left = examples[examples[:,best_attribute] < best_threshold]
        examples_right = examples[examples[:,best_attribute] >= best_threshold]

        tree.left_child = DTL(np.array(examples_left), attributes, tree.distribution, pruning_thr, classes)
        tree.right_child = DTL(np.array(examples_right), attributes, tree.distribution, pruning_thr, classes)
        return tree


def DTL_TopLevel(examples, pruning_thr, classes): # examples = training data with class
    attributes = range(len(examples[0]) - 1) # list from 0 - 15
    default = DISTRIBUTION(examples, classes)

    return DTL(examples, attributes, default, pruning_thr, classes)

def get_height(tree):
    if tree is None:
        return 0
    else:
        left = get_height(tree.left_child)
        right = get_height(tree.right_child)
        if left > right:
            return left + 1
        else:
            return right+1

index = 1
def print_tree(tree, height, num):
    if tree is None:
        return
    if height == 1:
        global index
        print("tree=" + "%2d"%int(num) + ", node=" + "%3d"%int(index) + ", feature=" + "%2d"%tree.attribute + ", thr=" + "%6.2f"%tree.threshold + ", gain=" + "%f"%tree.gain)
        index += 1
    elif height > 1:
        print_tree(tree.left_child, height-1, num)
        print_tree(tree.right_child, height-1, num)
            

def calculate(test, tree):
    if tree.left_child is None and tree.right_child is None:
        return tree.distribution
    elif tree.right_child is None:
        return calculate(test, tree.left_child)
    elif tree.left_child is None:
        return calculate(test, tree.right_child)
    else:
        if test[tree.attribute] < tree.threshold:
            return calculate(test, tree.left_child)
        else:
            return calculate(test, tree.right_child)


def decision_tree(training_data, test_data, option, pruning_thr):
    training = np.loadtxt(training_data)
    check_option(option)
    classes = np.array(training[:, -1])
    trees = [DTL_TopLevel(training, pruning_thr, classes)]

    if option == "forest3":
        for i in range(0, 2, 1):
            trees.append(DTL_TopLevel(training, pruning_thr, classes))
    elif option == "forest15":
        for i in range(0, 14, 1):
            trees.append(DTL_TopLevel(training, pruning_thr, classes))

    for i in range(len(trees)):
        height = get_height(trees[i])
        # print(height)
        for j in range(1, height+1):
            print_tree(trees[i], j, i+1)

    test = np.loadtxt(test_data)

    accuracy = []
    for i in range(len(test)):  
        this = -1
        if len(trees) == -1:
            dist = calculate(test[i], trees[0])
        else:
            avg = []
            for j in range(len(trees)):
                dist = calculate(test[i], trees[j])
                avg.append(np.argmax(dist))
            avg = np.array(avg)
            this = np.mean(avg) + min(classes)

        if int(this) == int(test[i][-1]):
            accuracy.append(1)
        else:
            accuracy.append(0)

        print("ID=" + "%5d"%int(i+1) + ", predicted=" + "%3d"%int(this) + ", true=" + "%3d"%int(test[i][-1]) + ", accuracy=" + "%4.2f"%int(accuracy[i]))

    accuracy = np.array(accuracy)
    print("classification accuracy=" + "%6.4f"%np.mean(accuracy) + "\n")


training_data = sys.argv[1]
test_data = sys.argv[2]
option = sys.argv[3]
pruning_thr = sys.argv[4]
index = 1


decision_tree(training_data, test_data, option, int(pruning_thr))