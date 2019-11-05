import numpy as np
import csv

mse = [[1.3], [2.3], [3.4]]
myArray = np.zeros(len(mse))

np.savetxt("testeq.txt", mse, delimiter=',')
result = np.genfromtxt('testeq.txt', delimiter=',')
result = np.reshape(result,(3,1))
