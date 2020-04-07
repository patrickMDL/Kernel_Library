import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand
import kernel_methods as kernel
import csv
import time

R = 50

linear_K = np.genfromtxt("/home/patrick/Desktop/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Polynomial_Kernel.txt",
    delimiter = ',')

gaussian_K = np.genfromtxt(
    "/home/patrick/Desktop/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Gaussian_Kernel.txt",
    delimiter=',')
cauchy_K = np.genfromtxt(
    "/home/patrick/Desktop/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Cauchy_Kernel.txt",
    delimiter = ',')

linear_K = np.reshape(linear_K, (3000,1))
gaussian_K = np.reshape(gaussian_K, (3000, 1))
cauchy_K = np.reshape(cauchy_K, (3000,1))

linear_K = linear_K / R
gaussian_K = gaussian_K / R
cauchy_K = cauchy_K/R

linear_K = 10 * np.log10(linear_K)
gaussian_K = 10 * np.log10(gaussian_K)
cauchy_K = 10 * np.log10(cauchy_K)

plt.plot(cauchy_K, color='green')
plt.plot(gaussian_K, color='red')
plt.plot(linear_K, color='blue')
plt.legend(( 'Cauchy', 'Gaussiano', 'Polinomial' ))

plt.xlim(0,2000)
plt.ylim(-43, -35)
plt.ylabel('MSE [dB]')
plt.xlabel('Iterações (n)')
plt.grid(True)
plt.savefig(
    '/home/patrick/Desktop/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE '
    'Gaussiano (Vermelho) vs Cauchy(Verde) vs Polinomial(Azul).eps',
    format='eps', dpi=300)
plt.show()
