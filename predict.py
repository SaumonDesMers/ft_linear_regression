import sys
import math
import pandas as pd


try:
    theta = pd.read_csv('theta.txt', sep=' ', header=None).to_numpy()[0]
    if math.isnan(theta[0]) or math.isnan(theta[1]):
        raise
except:
    print('Something went wrong with the theta.txt file!')
    exit()

print('Please entre the mileage:')

while True:

    try:
        txt = input('>> ')
    except:
        exit()

    if txt == 'exit':
        exit()
    elif not txt.isdigit():
        print('This is not a number!')
    else:
        print('A car with a {} km mileage should cost about {}!'.format(txt, theta[0] + int(txt) * theta[1]))