class LinearRegression:
    def __init__(self, n, features, target):
        self.n = n  #Learning rate
        self.bias = 0
        self.features = features
        self.target = target
        self.weights = []

    def predict(self):
        self.prediction = self.bias
        for i in range(len(self.features)): self.prediction += self.features[i]*self.weights[i]
        return self.prediction

    def initializeWeights(self):
        for i in range(len(self.features)):
            self.weights[i] = 0.5

    def gradiantDescendent():
        pass

    def errorCost(self):
        dim = len(self.target)
        errorSum = 0
        for i in range(dim - 1):
            errorSum += (self.target[i] - self.predict(i))**2

        return 1/(2*dim) * errorSum