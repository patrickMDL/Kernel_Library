import numpy as np
import numpy as np
import math 
import matplotlib.pyplot as plt
import random as rand
import kernel_methods as kernel 
import csv

R = 50
result = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Linear_Kernel.txt', delimiter=',')
result = np.reshape(result,((3000,1)))

mse = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Gaussian_Kernel.txt', delimiter=',')
mse = np.reshape(mse,((3000,1)))

MSE = mse / R
MSE = 10*np.log10(MSE)
RESULT = result/R
RESULT = 10*np.log10(RESULT)
plt.plot(MSE, color='red')
plt.plot(RESULT, color='blue')
plt.legend(('Gaussiano', 'Linear' ))
plt.ylabel('Mean Squared Error (MSE) [dB]')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Gaussiano (Red) vs Linear(blue).eps', format='eps', dpi=300)
plt.show()