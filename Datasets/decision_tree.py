
"""
Name: Ashwitha Kassetty
UTA ID: 1001551943
"""

import sys
import numpy as np
import pandas as pd
import math
import random

training_file = str(sys.argv[1])
test_file = str(sys.argv[2])
option = str(sys.argv[3])
pruning_thr = int(sys.argv[4])

td = pd.read_csv(training_file, delimiter= '\s+', header=None)
training_data = np.array(td)

examples = training_data
classes = np.array(examples[:, -1])
L = np.amin(classes)
H = np.amax(classes)
num_attributes = int(H - L + 1)


class Node:
    def __init__(self, attribute, threshold):
        self.attribute = attribute
        self.threshold = threshold
        self.gain = None
        self.distribution = None
        self.left_child = None
        self.right_child = None
        
        
def class_check(examples):
    for i in range(1, len(examples), 1):
        if examples[i][-1] != examples[i-1][-1]:
            return False
    return True

def distribution(examples):
    count = np.zeros(num_attributes)
    for i in range(0, len(examples), 1):
        count[int(examples[i][-1] - L)] += 1
    probability = np.array(count)/len(examples)
    return probability


def information_gain(examples, A, threshold):
    values = np.array(examples[:, A])
    class_distribution = distribution(examples)
    
    h_entropy = 0
    h_right = 0
    h_left = 0
    left = []
    right = []
    for i in range(0, len(class_distribution), 1):
        if class_distribution[i] == 0:
            h_entropy -= 0
        else:
            h_entropy -= class_distribution[i]*math.log2(class_distribution[i])
            
    for i in range(0, len(values), 1):
        if values[i] < threshold:a
            left.append(examples[i])
        else:
            right.append(examples[i])
    
    if len(left) == 0:
        h_left = 0
    else:
        l_distribution = distribution(np.array(left))
        for i in range(0, len(l_distribution), 1):
            if l_distribution[i] == 0:
                h_left -= 0
            else:
                h_left -= l_distribution[i]*math.log2(l_distribution[i])
                
    if len(right) == 0:
        h_right = 0
    else:
        r_distribution = distribution(np.array(right))
        for i in range(0, len(r_distribution), 1):
            if r_distribution[i] == 0:
                h_right -= 0
            else:
                h_right - r_distribution[i]*math.log2(r_distribution[i])
    
    h_left *= len(left)/len(values)
    h_right *= len(right)/len(values)
    
    gain = h_entropy - h_right - h_left
    
    return gain
                

def choose_attribute_optimized(examples, attributes):
    max_gain = -1
    best_attribute = -1 
    best_threshold = -1
    
    
    for A in range(0, len(attributes), 1):
#        print(A)
        attribute_values = np.array(examples[:,A])
        L = np.amin(attribute_values)
        M = np.amax(attribute_values)
        
        for K in range(1, 51, 1):
#            print(K)
            threshold = L + (K*((M-L)/51))
            gain = information_gain(examples, A, threshold)
            if gain > max_gain:
                max_gain = gain
                best_attribute = A
                best_threshold = threshold
    return best_attribute, best_threshold, max_gain
    
    
def choose_attribute_randomized(examples, attributes):
    max_gain = -1
    best_threshold = -1
    
    A = random.randint(attributes[0], attributes[-1]+1)
    attribute_values = np.array(examples[:,A])
    L = np.amin(attribute_values)
    M = np.amax(attribute_values)
    
    for K in range(1, 51, 1):
        threshold = L + (K*((M-L)/51))
        gain = information_gain(examples, A, threshold)
        if gain > max_gain:
            max_gain = gain
            best_threshold = threshold
    return A, best_threshold, max_gain


def choose_option(examples, attributes, option):
    if option == "optimized":
        return choose_attribute_optimized(examples, attributes)
    elif option == "randomized" or option == "forest3" or option == "forest15":
        return choose_attribute_randomized(examples, attributes)

def DTL(examples, attributes, default, pruning_threshold, option):
    print(examples)
    if len(examples) < pruning_threshold:
        x = Node(-1, -1)
        x.distribution = default
        x.gain = 0
        return x
    
    elif class_check(examples):
        x = Node(-1, -1)
        c = examples[0][-1]
        dist = np.zeros(num_attributes)
        dist[int(c - L)] = 1
        x.distribution = dist
        x.gain = 0
        return x
    
    else:
        best_attribute, best_threshold, max_gain = choose_option(examples, attributes, option)
        
        tree = Node(best_attribute, best_threshold)
        tree.gain = max_gain
        tree.distribution = distribution(examples)
        
        left_examples = []
        right_examples = []
        
        for i in range(0, len(examples), 1):
            if examples[i][best_attribute] < best_threshold:
                left_examples.append(examples[i])
            else:
                right_examples.append(examples[i])
        
        left_examples = np.array(left_examples)
        right_examples = np.array(right_examples)
        
        tree.left_child = DTL(left_examples, attributes, tree.distribution, pruning_threshold, option)
        tree.right_child = DTL(right_examples, attributes, tree.distribution, pruning_threshold, option)
        return tree
   
    
def DTL_Toplevel(examples, pruning_threshold, option):
    attributes = np.arange(0, len(examples[0])-1, 1)
    default = distribution(examples)
    return DTL(examples, attributes, default, pruning_threshold, option)


trees = []
tree = DTL_Toplevel(examples, pruning_thr, option)
#print(tree)
trees.append(tree)

if option == "forest3":
    for i in range(0, 2, 1):
        trees.append(DTL_Toplevel(examples, pruning_thr, option))
elif option == "forest15":
    for i in range(0, 14, 1):
        trees.append(DTL_Toplevel(examples, pruning_thr, option))
    
#print(trees)
def calculate_height(n):
    if n is None:
        return 0
    else:
        left_height = calculate_height(n.left_child)
        right_height = calculate_height(n.right_child)
        
        if left_height > right_height:
            return left_height+1
        else:
            return right_height+1

 
idx = 1
number = 1       
def printlevel(root, level):
    if root is None:
        return
    if level == 1:
        global idx
        global number
        print("tree:" + "%2d"%int(number) + ", node:" + "%3de"%int(idx) + ", feature:" + "%2d"%root.attribute + ", thr=" + "%6.2f"%root.threshold + ", gain=" + "%f"%root.gain)
        idx += 1
    elif level > 1:
        printlevel(root.left_child, level-1)
        printlevel(root.right_child, level-1)
        
                
def printorder(root):
    height = calculate_height(root)
    for i in range(1, height+1):
        printlevel(root, i)

for i in range(0, len(trees), 1):
    idx = 1
    printorder(trees[i])
    number += 1
   

ttd = pd.read_csv(test_file, delimiter= '\s+', header=None)
testing_data = np.array(ttd)


def check_accuracy(test, tr):
    if tr.left_child is None and tr.right_child is None:
        return tr.distribution
    elif tr.right_child is None:
        return check_accuracy(test, tr.left_child)
    elif tr.left_child is None:
        return check_accuracy(test, tr.right_child)
    else:
        if test[tr.attribute] < tr.threshotta=tld:
            return check_accuracy(test, tr.left_child)
        else:
            return check_accuracy(test, tr.right_child)
    

def calculate_accuracy(testing_data, trees):
    accuracy = []
    for i in range(0, len(testing_data), 1):
        c = -1
        if len(trees) == 1:
            ans = check_accuracy(testing_data[i], trees[0])
            c = np.argmax(ans) + L
        else:
            average = []
            for j in range(0, len(trees), 1):
                ans = check_accuracy(testing_data[i], trees[j])
                average.append(np.argmax(ans))
            average = np.array(average)
            c = np.mean(average) + L
            
        if int(c) == int(testing_data[i][-1]):
            accuracy.append(1)
        else:
            accuracy.append(0)
        
        print("ID=" + "%5d"%int(i+1) + ", predicted=" + "%3d"%int(c) + ", true=" + "%3d"%int(testing_data[i][-1]) + ", accuracy=" + "%4.2f"%int(accuracy[i]))
    return accuracy

        
        
accuracy = calculate_accuracy(testing_data, trees)
accuracy = np.array(accuracy)        
print("classification accuracy=" + "%6.4f"%np.mean(accuracy) + "\n")
            



 
    
    
