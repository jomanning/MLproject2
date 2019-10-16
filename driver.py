import numpy as np
import nearest_neighbors as nn
import clustering as clu
import perceptron as per

#writeup
nnTrainX1 = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
nnTrainY1 = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])
nnTestX1 = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
nnTestY1 = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])


#test
cluX1 = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
#writeup
cluX2 = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])

#test
perX1 = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
perY1 = np.array([[1], [1], [-1], [1], [-1], [-1], [-1]])
#writeup
perX2 = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
perY2 = np.array([[1], [1], [1], [-1], [-1], [-1]])

#test
#print('SELFTEST')
#print('ACC',nn.KNN_test(nnTrainX1, nnTrainY1, nnTrainX1, nnTrainY1, 1))

#writeup
print("K Nearest Neighbor")
nnAcc1 = nn.KNN_test(nnTrainX1, nnTrainY1, nnTestX1, nnTestY1, 1)
print("\tK=1","Accuracy",nnAcc1)
nnAcc2 = nn.KNN_test(nnTrainX1, nnTrainY1, nnTestX1, nnTestY1, 3)
print("\tK=3","Accuracy",nnAcc2)
nnAcc3 = nn.KNN_test(nnTrainX1, nnTrainY1, nnTestX1, nnTestY1, 5)
print("\tK=5","Accuracy",nnAcc3)
#writeup
nnK = nn.choose_K(nnTrainX1, nnTrainY1, nnTestX1, nnTestY1)
print("\tChoose K:",nnK)
print("\tAccuracy with chosen K:",nn.KNN_test(nnTrainX1, nnTrainY1, nnTestX1, nnTestY1,nnK))
print()

print("Clustering")
#test
C1 = clu.K_Means(cluX1, 3)
#print(C1)
#writeup
C2 = clu.K_Means(cluX2, 2)
print("\tK=2",C2)
C3 = clu.K_Means(cluX2, 3)
print("\tK=3",C3)
#writeup
CBetter1 = clu.K_Means_better(cluX2, 2)
print("\tBetter K=2",CBetter1)
CBetter2 = clu.K_Means_better(cluX2, 3)
print("\tBetter K=3",CBetter2)
print()

print("Perceptron")
#test
W_B1 = per.perceptron_train(perX1, perY1)
perAcc1 = per.perceptron_test(perX1, perY1, W_B1[0], W_B1[1])
#print("\tW",W_B1[0])
#print("\tB",W_B1[1])
#print("\tAccuracy", perAcc1)
#writeup
W_B2 = per.perceptron_train(perX2, perY2)
perAcc2 = per.perceptron_test(perX2, perY2, W_B2[0], W_B2[1])
print("\tW",W_B2[0])
print("\tB",W_B2[1])
print("\tAccuracy", perAcc2)
