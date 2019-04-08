#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

import utils
import math
import random
import copy
import numpy as np
from classifier import BinaryClassifier
from utils import read_data, build_vocab, vocab, get_feature_vectors, transform_data

from config import args

class NaiveBayes(BinaryClassifier):
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.dimension = args.vocab_size
        self.binary = args.bin_feats
        self.filepath = '../data/given/'
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        tr_x, tr_y = train_data
        self.x_dict_pos = np.zeros(self.dimension)        
        self.x_dict_neg = np.zeros(self.dimension)
        total = len(tr_x) - 1
        for i in range(0, total):
            if tr_y[i] == 1:
                self.add_pos(tr_x[i])
            else:
                self.add_neg(tr_x[i])
        for i in range(0, 3):
            self.x_dict_pos[i] = 1
            self.x_dict_neg[i] = 1
        self.x_dict_total = np.sum((self.x_dict_pos, self.x_dict_neg), axis=0)
        j = 0
        for i in np.nditer(self.x_dict_total):
            if i == 0:
                self.x_dict_total[j] = 1
            j = j + 1
        self.x_pos_prob = np.divide(self.x_dict_pos, self.x_dict_total)
        self.x_neg_prob = np.divide(self.x_dict_neg, self.x_dict_total)
    def add_pos(self, tr_xi):
        for i in transform_data(tr_xi):
            if i > 2 and i < self.dimension + 3:
                self.x_dict_pos[i] += 1
    def add_neg(self, tr_xi):
        for i in transform_data(tr_xi):
            if i > 2 and i < self.dimension + 3:
                self.x_dict_neg[i] += 1
    def predict(self, test_x):
        ret = []
        y = 0;
        for i in range(0, len(test_x) - 1):
            x = np.zeros(self.dimension)
            for i in transform_data(test_x[i]):
                if i > 2 and i < self.dimension + 3:
                    if self.binary:
                        x[i] = 1
                    else:
                        x[i] += 1
            y1 = 0
            y2 = 0
            pos = np.multiply(x, self.x_pos_prob)
            neg = np.multiply(x, self.x_neg_prob)
            for i in np.nditer(pos):
                if i > 0:
                    y1 += math.log(i, 2)
            for i in np.nditer(neg):
                if i > 0:
                    y2 += math.log(i, 2)
            if y1 >= y2:
                ret.append(1)
            else:
                ret.append(-1)
        return ret

