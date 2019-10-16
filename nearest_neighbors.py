import numpy as np

#takes in training data samples features X_train, training data samples labels Y_train, test data samples features X_test, test data labels Y_test,
#and number of neighbors to test on, K and returns the accuracy of the classifier on the test data
def KNN_test(X_train, Y_train, X_test, Y_test, K):
	if K == 0:
		return 0
	total = len(X_test)
	correct = 0
	#iterate through all the samples in test data
	for sample_idx, sample in enumerate(X_test):
		dists = []
		dist_idxs = []
		max_dist = 0
		max_idx = None
		#iterate through all the samples in train data & find the K closest to the test data sample
		for idx, trainData in enumerate(X_train):
			dist = 0
			#calculate the distance between each feature of the training data and test data, then square and sum them all up (no need for square root)
			for feat_idx in range(0,len(trainData)):
				dist += (sample[feat_idx] - trainData[feat_idx]) * (sample[feat_idx] - trainData[feat_idx])
			#if our list of nearest neighbors is less than K, add train data sample to nearest neighbors
			if len(dists) < K:
				dists += [dist]
				dist_idxs += [idx]
				if dist > max_dist:
					max_dist = dist
					max_idx = idx
			#otherwise see if training sample is closer than the furthest nearest neighbor and replace it
			elif dist < max_dist:
				dists.remove(max_dist)
				dist_idxs.remove(max_idx)
				dists += [dist]
				dist_idxs += [idx]
				max_dist = max(dists)
				max_idx = dist_idxs[dists.index(max_dist)]
		pos_votes = 0
		neg_votes = 0
		decision = None
		#iterate through the K nearest neighbors and calculate the majority vote
		for i in dist_idxs:
			if Y_train[i][0] == 1:
				pos_votes+=1
			else:
				neg_votes+=1
		if pos_votes > neg_votes:
			decision = 1
		else:
			decision = -1
		#if the classification matches the label add 1 to correct
		if decision == Y_test[sample_idx][0]:
			correct+=1
	return correct/total

#takes training data and validation data, and chooses the best number of nearest neighbors K that returns the highest accuracy on the validation data
def choose_K(X_train, Y_train, X_val, Y_val):
	best_acc = 0
	best_K = 0
	#iterate through all K from 1 to number of training data samples & return K with highest accuracy
	for i in range(1,len(X_train)+1):
		#if its even skip (K should be odd)
		if i%2 == 0:
			continue
		acc = KNN_test(X_train, Y_train, X_val, Y_val, i)
		if acc > best_acc:
			best_acc = acc
			best_K = i
	return best_K
