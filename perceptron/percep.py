import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Perceptron:
  
    def __init__ (self):
        self.w = None #weight
        self.b = None #bias
    
    #if sum(Wi * Xi) >= threshhold return 1, 0 otherwise for every element in X
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    #excutes perceptron learning algorthim
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
    #trains set X consiting of class Y for # epochs w/ learning rate lr
    def fit(self, X, Y, epochs = 1, lr = 1):
        #self.w = np.ones(X.shape[1])
        self.w = np.random.uniform(-1.0,1.0,X.shape[1]) #random vector of weights
        self.b = 0

        accuracy = {}
        max_accuracy = 0

        wt_matrix = []

        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)    
            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                j = i
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        print("Max Accuracy, At Epoch #")
        print(max_accuracy,j)
        #print(accuracy.values())
        plt.plot(list(accuracy.values()))
        plt.xlabel("epoch #")
        plt.ylabel("accuracy")
        plt.ylim([0, 1])
        plt.show()

        return np.array(wt_matrix)
