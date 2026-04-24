from linearRegression import LinearRegression as lr

features = [
    [1],
    [2],
    [3],
    [4],
    [5]
]

target = [3, 5, 7, 9, 11]

model = lr(learningRate=0.05, 
           features=features, 
           target=target)

errorCost = model.errorCost()
print(errorCost)

model.gradiantDescent()

errorCost = model.errorCost()
print(errorCost)
print(model.weights)
print(model.bias)
print(model.predict([6]))