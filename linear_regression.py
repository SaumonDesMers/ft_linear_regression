import matplotlib.pyplot as plt
import numpy as np
import pandas


data = pandas.read_csv('data.csv').to_numpy().transpose()


# y = theta1 * x + theta0
theta0 = 0
theta1 = 0
def estime_price(km):
    return theta1 * km + theta0


# plot estimate
minKm = np.min(data[0])
maxKm = np.max(data[0])
minKmPrice = estime_price(minKm)
maxKmPrice = estime_price(maxKm)

plt.plot([minKm, maxKm], [minKmPrice, maxKmPrice], 'r')


# plot data
plt.plot(data[0], data[1], 'bo')


# error
data = data.transpose()
for km, price in data:
    plt.plot([km, km], [price, estime_price(km)], '--m')


plt.xlabel('km')
plt.ylabel('price')
plt.show()
