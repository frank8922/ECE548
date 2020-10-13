import numpy as np
from numpy.core.fromnumeric import mean
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
    def fit(self, X, Y,XTest,YTest, epochs = 1, lr = 1):
        #self.w = np.ones(X.shape[1])
        self.w = np.random.uniform(-1.0,1.0,X.shape[1]) #random vector of weights
        self.b = 0

        accuracy = []
        max_accuracy = 0

        wt_matrix = []
        testAccuracy = []
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
            accuracy.append(accuracy_score(self.predict(X), Y))
            testAccuracy.append(accuracy_score(self.predict(XTest), YTest))
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                j = i
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb
        accuracy = np.array(accuracy)
        error = 1 - accuracy
        plt.style.use('seaborn')
        plt.plot(error)
        plt.xlabel('# of epochs')
        plt.ylabel('Error rate')
        plt.show()
        print("Max Accuracy, At Epoch #")
        print(max_accuracy,j)
        # print(accuracy)
        #print(accuracy.values())
        plt.plot(accuracy)
        plt.xlabel("epoch #")
        plt.ylabel("accuracy")
        plt.ylim([0, 1])
        # plt.show()

        plt.plot(testAccuracy)
        # plt.xlabel("epoch #")
        # plt.ylabel("accuracy")
        # plt.ylim([0, 1])
        plt.legend(['training accuracy','testing accuracy'])
        plt.show()

        return np.array(wt_matrix)