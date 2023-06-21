import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas


matplotlib.rcParams['mathtext.fontset'] = 'cm'


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


def plot_graph(ax, X, Y, t0, t1):

    ax.set_title(r'$\theta_0 = {:f} \quad \theta_1 = {:f}$'.format(t0, t1))

    # plot prediction
    minX = min(X)
    maxX = max(X)
    minX_prediction = t0 + minX * t1
    maxX_prediction = t0 + maxX * t1

    ax.plot([minX, maxX], [minX_prediction, maxX_prediction], 'r')

    # plot data
    ax.plot(X, Y, 'bo')

    # plot error
    for i in range(X.shape[0]):
        ax.plot([X[i], X[i]], [Y[i], t0 + X[i] * t1], '--m')



if __name__ == '__main__':

    # read data
    data = pandas.read_csv('data.csv').to_numpy().transpose()

    # normalize data
    X = normalize(data[0])
    Y = normalize(data[1])

    print('*** Original theta ***')
    print('theta0 = {:<f}  theta1 = {:<f}'.format(theta0, theta1))

    # learn loop
    prev_MSE = MSE(X, Y)
    mse = [prev_MSE]
    for i in range(10000):
        learn(X, Y)
        mse.append(MSE(X, Y))

    print('\n*** Final theta on normalize data ***')
    print('theta0 = {:<f}  theta1 = {:<f}'.format(theta0, theta1))


    # create plot
    fig, axs = plt.subplots(1, 2)

    # get theta for original data
    minX = min(data[0])
    maxX = max(data[0])
    minY = min(data[1])
    maxY = max(data[1])

    _theta0 = (maxY - minY) * (theta0 - minX * theta1 / (maxX - minX)) + minY
    _theta1 = theta1 * (maxY - minY) / (maxX - minX)

    # plot graphe on original data
    axs[0].set_xlabel('mileage')
    axs[0].set_ylabel('price')
    plot_graph(axs[0], data[0], data[1], _theta0, _theta1)

    print('\n*** Final theta on original data ***')
    print('theta0 = {:<f}  theta1 = {:<f}'.format(_theta0, _theta1))

    # write theta value to file
    f = open('theta.txt', 'w')
    f.write('{} {}'.format(_theta0, _theta1))
    f.close()

    # plot MSE value
    axs[1].set_title(r'$MSE = \frac{1}{m} \sum_{i=0}^m (Y - \^Y)^2$')
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('MSE')
    axs[1].plot(mse)

    fig.set_figwidth(15)
    plt.show()
