import numpy as np

def perceptron_train(X, Y):
	w = np.array([0])
	b = np.array([0])
	return np.array([w,b])

def perceptron_test(X_test, Y_test, w, b):
	acc = 0
	return acc