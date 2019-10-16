import numpy as np

#takes training data features X and labels Y, and returns [w,b], a list containing a list of weights, w and the bias, b.
def perceptron_train(X, Y):
	#initialize w to the 0 vector and b to 0
	w = [0] * len(X[0])
	b = 0
	updated = True
	#run loop until it can go through all the samples without updating w and b
	while updated:
		updated = False
		#for each sample, calculate a
		for sample_idx, x in enumerate(X):
			a = b
			for feat_idx, feat in enumerate(x):
				a += w[feat_idx] * feat
			#if a and Y aren't the same sign, then update the weights and bias
			if a * Y[sample_idx][0] <= 0:
				updated = True
				for feat_idx, feat in enumerate(x):
					w[feat_idx] += feat * Y[sample_idx][0]
				b += Y[sample_idx][0]
	return [w,b]

#takes test data features X_test and labels Y_test, weight vector w, and bias and returns the accuracy of the perceptron on the test data
def perceptron_test(X_test, Y_test, w, b):
	total = len(X_test)
	correct = 0
	#for each sample in the test data, calculate a
	for idx, x in enumerate(X_test):
		decision = None
		a = b
		for feat_idx,feat in enumerate(x):
			a += w[feat_idx] * feat
		#if a > 0, classify it a 1, otherwise classify it a -1
		if a > 0:
			decision = 1
		else:
			decision = -1
		#if the classification is the same as the label increment number of correct
		if decision == Y_test[idx][0]:
			correct+=1
	return correct/total
