#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
import utils
import random
import copy
import numpy as np
from classifier import BinaryClassifier
from utils import read_data, build_vocab, vocab, get_feature_vectors, transform_data

from config import args

class Perceptron(BinaryClassifier):
    
    def __init__(self, args):
        self.dimension = args.f_dim
        self.iteration = args.num_iter
        self.rate = args.lr
        self.binary = args.bin_feats
        self.filepath = '../data/given/'
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        #raise NotImplementedError
        tr_x, tr_y = train_data
        self.w = np.zeros(self.dimension)
        self.b = 0.0
        self.x_dict = copy.deepcopy(vocab) 
        self.x_dict = {x:0 for x in self.x_dict}
        #print(x_dict)
        total = len(tr_x) - 1
        random.seed(0)
        #self.cal_new_w(tr_x[0], tr_y[0]) 
        for i in range(0, self.iteration):
            j = total
            t_x = copy.deepcopy(tr_x)
            t_y = copy.deepcopy(tr_y)
            for i in range(0, total):
                j = j - 1
                random.seed(j)
                k = random.randint(0, j)
                w = self.cal_new_w(t_x.pop(k), t_y.pop(k)) 
        
    def cal_new_w(self, tr_xi, label):
        x = np.zeros(self.dimension)
        for i in transform_data(tr_xi):
            if i > 2 and i < self.dimension + 3:
                if self.binary:
                    x[i - 3] = 1
                else:
                    x[i - 3] += 1
        self.b = self.b + self.rate * label
        y = np.inner(self.w, x) + self.b
        x = x * label * self.rate
        if (label == 1 and y < 0) or (label == -1 and y >= 0):
            self.w = np.sum((self.w, x), axis=0)

        
    def predict(self, test_x):
        ret = []
        y = 0;
        for i in range(0, len(test_x)):
            x = np.zeros(self.dimension)
            for i in transform_data(test_x[i]):
                if i > 2 and i < self.dimension + 3:
                    if self.binary:
                        x[i - 3] = 1
                    else:
                        x[i - 3] += 1
            y = np.inner(self.w, x) + self.b
            if y >= 0:
                ret.append(1)
            else:
                ret.append(-1)
        return ret
class AveragedPerceptron(BinaryClassifier):
    
    def __init__(self, args):
        self.dimension = args.f_dim
        self.iteration = args.num_iter
        self.rate = args.lr
        self.binary = args.bin_feats
        self.filepath = '../data/given/'
                
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        tr_x, tr_y = train_data
        self.w = np.zeros(self.dimension)
        self.b = 0
        self.x_dict = copy.deepcopy(vocab) 
        self.x_dict = {x:0 for x in self.x_dict}
        #print(x_dict)
        total = len(tr_x) - 1
        self.survival = 1
        random.seed(2)
        for i in range(0, self.iteration):
            j = total
            t_x = copy.deepcopy(tr_x)
            t_y = copy.deepcopy(tr_y)
            for i in range(0, total):
                j = j - 1
                random.seed(j)
                k = random.randint(0, j)
                w = self.cal_new_w(t_x.pop(k), t_y.pop(k)) 

    def cal_new_w(self, tr_xi, label):
        x = np.zeros(self.dimension)
        for i in transform_data(tr_xi):
            if i > 2 and i < self.dimension + 3:
                if self.binary:
                    x[i - 3] = 1
                else:
                    x[i - 3] += 1
        
        y = np.inner(self.w, x) + self.b
        x = x * float(label) * self.rate
        if (label == 1 and y >= 0) or (label == -1 and y < 0):
            self.survival = self.survival + 1
        else:
            self.b = self.b + self.rate * label / (self.survival + 1)
            tw = np.sum((self.w, x), axis=0)
            self.w = self.w * self.survival
            self.w = np.sum((self.w, tw), axis=0)
            self.w = self.w / (self.survival + 1)
            self.survival = 1
    def predict(self, test_x):
        ret = []
        y = 0;
        for i in range(0, len(test_x)):
            x = np.zeros(self.dimension)
            for i in transform_data(test_x[i]):
                if i > 2 and i < self.dimension + 3:
                    if self.binary:
                        x[i - 3] = 1
                    else:
                        x[i - 3] += 1

            y = np.inner(self.w, x) + self.b
            if y >= 0:
                ret.append(1)
            else:
                ret.append(-1)
        return ret