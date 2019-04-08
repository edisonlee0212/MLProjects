##############
# Name: Bosheng Li
# email: li2343@purdue.edu
# Date: 3/4/2019

import numpy as np
import sys
import os
import pandas as pd
import copy
import time
columns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'relatives', 'IsAlone']
major_label = 0
min_split = 0
def entropy(freqs):
	all_freq = sum(freqs)
	entropy = 0 
	for fq in freqs:
		prob = fq * 1.0 / all_freq
		if abs(prob) > 1e-8:
			entropy += -prob * np.log2(prob)
	return entropy

def infor_gain(before_split_freqs, after_split_freqs):
	gain = entropy(before_split_freqs)
	overall_size = sum(before_split_freqs)
	for freq in after_split_freqs:
		ratio = sum(freq) * 1.0 / overall_size
		gain -= ratio * entropy(freq)
	return gain

class Node(object):
	def __init__(self):
		self.notLeaf = True
		self.label = None
		self.attribute = None
		self.mid_point = None
		self.r = None
		self.l = None

	def setLeaf(self, label):
		self.notLeaf = False
		self.label = label

	def test_split(self, index, value, data):

		left, right = [], []

		for row in data:
			if row[index] <= value:
				left.append(row)
			else:
				right.append(row)

		return np.asarray(left), np.asarray(right)


	def cal_infor_gain(self, data, index):
		max_infor_gain = 0.0
		ret_mid_point = -1
		best_left_data = None
		best_right_data = None

		#Iterate through ecery possible midpoints and select the best one.
		#Noted that I did not use sort for this. Instead I use the unique function which saves a lot of time.
		for value in np.unique(data[:, index]):
			tmp_left_data, tmp_right_data = self.test_split(index, value, data)
			left_survived = 0
			if tmp_left_data.size and tmp_right_data.size:
				left_total = tmp_left_data.shape[0]
				right_survived = 0
				right_total = tmp_right_data.shape[0]
				for label in tmp_left_data[:, 7]:
					if label == 1:
						left_survived = left_survived + 1
				for label in tmp_right_data[:, 7]:
					if label == 1:
						right_survived = right_survived + 1				
				tmp_infor_gain = infor_gain([left_total+right_total - left_survived - right_survived, left_survived+right_survived], [[left_total - left_survived, left_survived],[right_total - right_survived, right_survived]])
				if tmp_infor_gain > max_infor_gain:
					max_infor_gain = tmp_infor_gain
					ret_mid_point = value
					best_left_data = tmp_left_data
					best_right_data = tmp_right_data
		return max_infor_gain, ret_mid_point, best_left_data, best_right_data

	def major_vote(self, major, data):
		error = 0
		for index in range(len(data)):
			if major != data[index]:
				error += 1
		return float(error)

	def vote(self, data):
		survived = 0
		died = 0
		major = 0
		for label in data[:, 7]:
			if label == 1:
				survived = survived + 1
			else:
				died = died + 1
		if major_label == 0:
			if died >= survived:
				major = 0
			else:
				major = 1
		else:
			if died > survived:
				major = 0
			else:
				major = 1
		return major

	def prune(self, validation_data):
		if validation_data.shape[0] == 0:
			return self

		left_data, right_data = self.test_split(columns.index(self.attribute), self.mid_point, validation_data)

		if self.l.notLeaf:
			self.l = self.l.prune(left_data)

		if self.r.notLeaf:
			self.r = self.r.prune(right_data)
		if self.l and self.r and not self.l.notLeaf and not self.r.notLeaf:
			if left_data.shape[0] == 0:
				left_sum = 0
			else:
				left_sum = self.major_vote(self.l.label, left_data[:, -1])

			if right_data.shape[0] == 0:
				right_sum = 0
			else:
				right_sum = self.major_vote(self.r.label, right_data[:, -1])

			error_no_merge = pow(left_sum, 2) + pow(right_sum, 2)
			tree_mean = self.vote(validation_data)
			error_merge = pow(self.major_vote(tree_mean, validation_data[:, -1]), 2)
			if error_merge < error_no_merge:
				new_node = Node()
				new_node.setLeaf(tree_mean)
				return new_node
			else:
				return self
		return self

	def ID3(self, data, max_depth):
		#If all examples are have same label
		same_label = True
		old_label = data[0, 7]
		for label in data[:, 7]:
			if old_label != label:
				same_label = False
				break
			old_label = label

		if same_label:
			self.setLeaf(label)
			return self

		row_size = data.shape[0]
		if row_size < min_split:
			self.setLeaf(self.vote(data))
			return self

		if max_depth == 0:
			survived = 0
			died = 0
			self.setLeaf(self.vote(data))
			return self

		best_mid_point = 0.0
		best_infor_gain = 0.0
		best_index = -1
		#Iterate through each attributes and select the best split
		for index in range(len(columns)):
			infor_gain, tmp_mid_point, tmp_left_data, tmp_right_data = self.cal_infor_gain(data, index)
			if infor_gain > best_infor_gain:
				best_infor_gain = infor_gain
				best_mid_point = tmp_mid_point
				best_index = index
				left_data = tmp_left_data
				right_data = tmp_right_data
		if best_infor_gain == 0.0:
			self.setLeaf(major_label)
			return self
		self.attribute = columns[best_index]
		self.mid_point = best_mid_point

		#Create branches and build the tree continuously with recursion with splited data
		self.l = Node()
		self.l.ID3(left_data, max_depth - 1)
		self.r = Node()
		self.r.ID3(right_data, max_depth - 1)

	def count_nodes(self):
		if self.notLeaf == False:
			return 1
		left = 0
		right = 0
		if self.l:
			left = self.l.count_nodes()
		if self.r:
			right = self.r.count_nodes()
		return left + right + 1

class Tree(object):
	def __init__(self, train_folder, test_folder, model):
		self.train_folder = train_folder
		self.test_folder = test_folder
		self.model = model
		self.root = None

	def load_train_data(self):
		train_data_file = self.train_folder + "/titanic-train.data"
		train_label_file = self.train_folder + "/titanic-train.label"
		train_data = pd.read_csv(train_data_file, delimiter = ',', index_col=None, engine='python')
		columns = list(train_data.columns.values.tolist())
		train_label = pd.read_csv(train_label_file, delimiter = ',', index_col=None, engine='python')
		data = pd.concat([train_data, train_label], axis=1)
		data_matrix = data.as_matrix()
		return data_matrix

	def load_test_data(self):
		test_data_file = test_folder + "/titanic-test.data"
		test_label_file = test_folder + "/titanic-test.label"
		test_data = pd.read_csv(test_data_file, delimiter = ',', index_col=None, engine='python')
		test_label = pd.read_csv(test_label_file, delimiter = ',', index_col=None, engine='python')
		data = pd.concat([test_data, test_label], axis=1)
		data_matrix = data.as_matrix()
		return data_matrix

	def build_tree(self, data, max_depth):
		self.root = Node()
		self.root.ID3(data, max_depth)
		return self.root

	def prune_tree(self, validation_data):
		if validation_data.shape[0] == 0:
			return self
		self.root.prune(validation_data)
		return self

	def predict(self, row):
		ret = 0
		node = self.root
		while node.notLeaf:
			if node.attribute == "Pclass":
				if row[0] <= node.mid_point:
					node = node.l
				else:
					node = node.r
			elif node.attribute == "Sex":
				if row[1] <= node.mid_point:
					node = node.l
				else:
					node = node.r
			elif node.attribute == "Age":
				if row[2] <= node.mid_point:
					node = node.l
				else:
					node = node.r
			elif node.attribute == "Fare":
				if row[3] <= node.mid_point:
					node = node.l
				else:
					node = node.r
			elif node.attribute == "Embarked":
				if row[4] <= node.mid_point:
					node = node.l
				else:
					node = node.r
			elif node.attribute == "relatives":
				if row[5] <= node.mid_point:
					node = node.l
				else:
					node = node.r
			elif node.attribute == "IsAlone":
				if row[6] <= node.mid_point:
					node = node.l
				else:
					node = node.r
		return node.label

	def accuracy_metric(self, actual, predicted):
		correct = 0

		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0

if __name__ == "__main__":
		# parse arguments
		train_folder = sys.argv[1]
		test_folder = sys.argv[2]
		model = sys.argv[3]
		train_set_percentage = sys.argv[4]
		train_data_file = train_folder + "/titanic-train.data"
		train_label_file = train_folder + "/titanic-train.label"
		test_data_file = test_folder + "/titanic-test.data"
		test_label_file = test_folder + "/titanic-test.label"
		#print("train data: ", train_data_file)
		#print("train label: ", train_label_file)
		#print("test data: ", test_data_file)
		#print("test label: ", test_label_file)
		#print("model: ", model)
		#print("train set train_set_percentage: ", train_set_percentage)
		train_data = pd.read_csv(train_data_file, delimiter = ',', index_col=None, engine='python')
		test_data = pd.read_csv(test_data_file, delimiter = ',', index_col=None, engine='python')
		train_label = pd.read_csv(train_label_file, delimiter = ',', index_col=None, engine='python')
		test_label = pd.read_csv(test_label_file, delimiter = ',', index_col=None, engine='python')
		t = Tree(train_folder, test_folder, model)
		train = t.load_train_data()
		test = t.load_test_data()
		if model == "vanilla":
			new_train = copy.deepcopy(train)
			new_train = new_train[0:int(len(train) * int(train_set_percentage) / 100), :]
			new_test = copy.deepcopy(test)
			max_depth = float("inf")
			t.build_tree(new_train, max_depth)


			#UNCOMMENT HERE TO GET THE NODE AMOUNT OF THE TREE
			#print("Node amount: %d" % t.root.count_nodes())



			perdict_train = list()
			for row in new_train:
				train_row_perdict = t.predict(row)
				perdict_train.append(train_row_perdict)
			#print(perdict_train)
			accuracy = t.accuracy_metric(new_train[:, -1], perdict_train) / 100
			print("Train set accuracy: %.4f" % accuracy)

			perdict_test = list()
			for row in new_test:
				row_perdict = t.predict(row)
				perdict_test.append(row_perdict)
			#print(perdict_test)
			accuracy = t.accuracy_metric(new_test[:, -1], perdict_test) / 100
			print("Test set accuracy: %.4f" % accuracy)

		elif model == "depth":
			validation_set_percentage = sys.argv[5]
			max_depth = int(sys.argv[6])
			new_train = copy.deepcopy(train)
			new_train = new_train[0:int(len(train) * int(train_set_percentage) / 100), :]
			new_validation = copy.deepcopy(train)
			new_validation = new_validation[int(len(new_validation) * (100-int(validation_set_percentage)) / 100):, :]
			new_test = copy.deepcopy(test)
			
			t.build_tree(new_train, max_depth)



			#UNCOMMENT HERE TO GET THE NODE AMOUNT OF THE TREE
			#print("Node amount: %d" % t.root.count_nodes())



			perdict_train = list()
			for row in new_train:
				train_row_perdict = t.predict(row)
				perdict_train.append(train_row_perdict)
			#print(perdict_train)
			accuracy = t.accuracy_metric(new_train[:, -1], perdict_train) / 100
			print("Train set accuracy: %.4f" % accuracy)

			perdict_validation = list()
			for row in new_validation:
				validation_row_perdict = t.predict(row)
				perdict_validation.append(validation_row_perdict)
			#print(perdict_train)
			accuracy = t.accuracy_metric(new_validation[:, -1], perdict_validation) / 100
			print("Validation set accuracy: %.4f" % accuracy)

			perdict_test = list()
			for row in new_test:
				row_perdict = t.predict(row)
				perdict_test.append(row_perdict)
			#print(perdict_test)
			accuracy = t.accuracy_metric(new_test[:, -1], perdict_test) / 100
			print("Test set accuracy: %.4f" % accuracy)

		elif model == "min_split":
			validation_set_percentage = sys.argv[5]
			min_split = int(sys.argv[6])
			new_train = copy.deepcopy(train)
			new_train = new_train[0:int(len(train) * int(train_set_percentage) / 100), :]
			new_validation = copy.deepcopy(train)
			new_validation = new_validation[int(len(new_validation) * (100-int(validation_set_percentage)) / 100):, :]
			new_test = copy.deepcopy(test)
			max_depth = float("inf")

			t.build_tree(new_train, max_depth)

			

			#UNCOMMENT HERE TO GET THE NODE AMOUNT OF THE TREE
			#print("Node amount: %d" % t.root.count_nodes())



			perdict_train = list()
			for row in new_train:
				train_row_perdict = t.predict(row)
				perdict_train.append(train_row_perdict)
			#print(perdict_train)
			accuracy = t.accuracy_metric(new_train[:, -1], perdict_train) / 100
			print("Train set accuracy: %.4f" % accuracy)

			perdict_validation = list()
			for row in new_validation:
				validation_row_perdict = t.predict(row)
				perdict_validation.append(validation_row_perdict)
			#print(perdict_train)
			accuracy = t.accuracy_metric(new_validation[:, -1], perdict_validation) / 100
			print("Validation set accuracy: %.4f" % accuracy)

			perdict_test = list()
			for row in new_test:
				row_perdict = t.predict(row)
				perdict_test.append(row_perdict)
			#print(perdict_test)
			accuracy = t.accuracy_metric(new_test[:, -1], perdict_test) / 100
			print("Test set accuracy: %.4f" % accuracy)

		elif model == "prune":
			validation_set_percentage = sys.argv[5]
			new_train = copy.deepcopy(train)
			new_train = new_train[0:int(len(train) * int(train_set_percentage) / 100), :]
			new_validation = copy.deepcopy(train)
			new_validation = new_validation[int(len(new_validation) * (100-int(validation_set_percentage)) / 100):, :]
			new_test = copy.deepcopy(test)
			max_depth = float("inf")

			t.build_tree(new_train, max_depth)
			t.prune_tree(new_validation)

			

			#UNCOMMENT HERE TO GET THE NODE AMOUNT OF THE TREE
			#print("Node amount: %d" % t.root.count_nodes())



			perdict_train = list()
			for row in new_train:
				train_row_perdict = t.predict(row)
				perdict_train.append(train_row_perdict)
			#print(perdict_train)
			accuracy = t.accuracy_metric(new_train[:, -1], perdict_train) / 100
			print("Train set accuracy: %.4f" % accuracy)

			perdict_test = list()
			for row in new_test:
				row_perdict = t.predict(row)
				perdict_test.append(row_perdict)
			#print(perdict_test)
			accuracy = t.accuracy_metric(new_test[:, -1], perdict_test) / 100
			print("Test set accuracy: %.4f" % accuracy)

		else:
			print("Usage: python ID3.py")
	# build decision tree
	# predict on testing set & evaluate the testing accuracy
	
