import numpy as np
import pandas


# prediction function
# y = theta0 + theta1 * x
theta0 = 0
theta1 = 0
def predict(x):
    return theta0 + theta1 * x


# normalize data
def normalize(data):
    _min = min(data)
    diff = max(data) - _min
    return np.array([(x - _min) / diff for x in data])


# Mean Squared Error
def MSE(X, Y):
    return sum([(predict(X[i]) - Y[i])**2 for i in range(X.shape[0])]) / X.shape[0]


# function that improve theta with the MSE partial derivative with respect to theta0 and theta1
def learn(X, Y):
    global theta0, theta1
    learning_rate = 0.005

    theta0_gradient = (sum([predict(X[i]) - Y[i]
                       for i in range(X.shape[0])]) / X.shape[0])
    theta1_gradient = (sum([(predict(X[i]) - Y[i]) * X[i]
                       for i in range(X.shape[0])]) / X.shape[0])

    theta0 -= learning_rate * theta0_gradient
    theta1 -= learning_rate * theta1_gradient


if __name__ == '__main__':

    # read data
    data = pandas.read_csv('data.csv').to_numpy().transpose()

    # normalize data
    X = normalize(data[0])
    Y = normalize(data[1])

    # learn loop
    print('*** Original theta ***')
    print('theta0 = {:<f}  theta1 = {:<f}'.format(theta0, theta1))

    for i in range(10000):
        learn(X, Y)

    print('\n*** Final theta on normalize data ***')
    print('theta0 = {:<f}  theta1 = {:<f}'.format(theta0, theta1))


    # get theta for original data (denormalizing)
    minX = min(data[0])
    maxX = max(data[0])
    minY = min(data[1])
    maxY = max(data[1])

    _theta0 = (maxY - minY) * (theta0 - minX * theta1 / (maxX - minX)) + minY
    _theta1 = theta1 * (maxY - minY) / (maxX - minX)

    print('\n*** Final theta on original data ***')
    print('theta0 = {:<f}  theta1 = {:<f}'.format(_theta0, _theta1))


    # write theta value to file
    f = open('theta.txt', 'w')
    f.write('{} {}'.format(_theta0, _theta1))
    f.close()
