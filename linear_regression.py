import matplotlib.pyplot as plt
import numpy as np
import pandas


# prediction function
# y = theta0 + theta1 * x
theta0 = 0
theta1 = 0


def estime_price(x):
    return theta0 + theta1 * x


# normalize data
def normalize(data):
    _min = min(data)
    diff = max(data) - _min
    return np.array([(x - _min) / diff for x in data])


# Mean Squared Error
def MSE(X, Y):
    return sum([(estime_price(X[i]) - Y[i])**2 for i in range(X.shape[0])]) / X.shape[0]


# function that improve theta with the MSE partial derivative with respect to theta0 and theta1
def learn(X, Y):
    global theta0, theta1
    learning_rate = 0.0005

    theta0_gradient = (sum([estime_price(X[i]) - Y[i]
                       for i in range(X.shape[0])]) / X.shape[0])
    theta1_gradient = (sum([(estime_price(X[i]) - Y[i]) * X[i]
                       for i in range(X.shape[0])]) / X.shape[0])

    theta0 -= learning_rate * theta0_gradient
    theta1 -= learning_rate * theta1_gradient


# read data
data = pandas.read_csv('data.csv').to_numpy().transpose()

# normalize data
X = normalize(data[0])
Y = normalize(data[1])

# learn loop
mse = [MSE(X, Y)]
print('theta0 = {:<30}  theta1 = {:<30}  MSE = {}'.format(
    theta0, theta1, MSE(X, Y)))
for i in range(100000):
    learn(X, Y)
    mse.append(MSE(X, Y))
print('theta0 = {:<30}  theta1 = {:<30}  MSE = {}'.format(
    theta0, theta1, MSE(X, Y)))


# create plot
fig, axs = plt.subplots(2)

# plot estimate
minKm = np.min(X)
maxKm = np.max(X)
minKmPrice = estime_price(minKm)
maxKmPrice = estime_price(maxKm)

axs[0].plot([minKm, maxKm], [minKmPrice, maxKmPrice], 'r')


# plot data
axs[0].plot(X, Y, 'bo')


# error
for i in range(X.shape[0]):
    axs[0].plot([X[i], X[i]], [Y[i], estime_price(X[i])], '--m')

# plot MSE value
axs[1].plot(mse)


# axs[0].xlabel('km')
# axs[0].ylabel('price')
plt.show()
