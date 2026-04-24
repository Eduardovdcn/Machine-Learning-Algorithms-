class LinearRegression:
    def __init__(self, learningRate, features, target):
        self.learningRate = learningRate  
        self.numIterations = 100
        self.features = features    #EX: [[0,1,1], [0,4,2]] -> Size x Dim
        self.size = len(self.features)   #Number of samples
        self.dim = len(self.features[0])   #Number of dim
        self.target = target #[0,1,3,6,4] -> M size
        self.weights = [0.5] * self.dim
        self.bias = 0


    def predict(self, sample):
        prediction = self.bias
        for i in range(self.dim): prediction += sample[i] * self.weights[i]
        return prediction


    def gradiantDescent(self):
        for _ in range(self.numIterations):
            weightGradient = [0.0] * self.dim
            biasGradient = 0.0

            for sample in range(self.size):
                error = (self.predict(self.features[sample]) - self.target[sample])
                for dim in range(self.dim):                   
                    weightGradient[dim] += error * self.features[sample][dim]

                biasGradient += error

            for dim in range(self.dim):
                weightGradient[dim] /= self.size
                self.weights[dim] -= self.learningRate * weightGradient[dim]

            biasGradient /= self.size
            self.bias -= self.learningRate * biasGradient

    def errorCost(self):
        
        errorSum = 0
        for i in range(self.dim):
            errorSum += (self.target[i] - self.predict(self.features[i]))**2

        return 1/(2*self.dim) * errorSum