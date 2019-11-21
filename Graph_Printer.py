import numpy as np
import math 
import matplotlib.pyplot as plt
import random as rand
import kernel_methods as kernel 
import csv

R = 50

kernel_Linear = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Linear_Kernel.txt', delimiter=',')
kernel_Linear = np.reshape(kernel_Linear,(3000,1))

kernel_Polynomial = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Polynomial_Kernel.txt', delimiter=',')
kernel_Polynomial = np.reshape(kernel_Polynomial,(3000,1))

kernel_Gaussiano = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Gaussian_Kernel.txt', delimiter=',')
kernel_Gaussiano = np.reshape(kernel_Gaussiano,(3000,1))

kernel_Exponential = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Exponential_Kernel.txt', delimiter=',')
kernel_Exponential = np.reshape(kernel_Exponential,(3000,1))

kernel_Laplacian = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Laplacian_Kernel.txt', delimiter=',')
kernel_Laplacian = np.reshape(kernel_Laplacian,(3000,1))

kernel_Sigmoid = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Sigmoid_Kernel.txt', delimiter=',')
kernel_Sigmoid = np.reshape(kernel_Sigmoid,(3000,1))

kernel_Rational_Quadratic = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Rational_Quadratic_Kernel.txt', delimiter=',')
kernel_Rational_Quadratic = np.reshape(kernel_Rational_Quadratic,(3000,1))

kernel_Multiquadric = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Multiquadratic_Kernel.txt', delimiter=',')
kernel_Multiquadric = np.reshape(kernel_Multiquadric,(3000,1))

kernel_Inverse_Multiquadric = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Inverse_Multiquadratic_Kernel.txt', delimiter=',')
kernel_Inverse_Multiquadric = np.reshape(kernel_Inverse_Multiquadric,(3000,1))

kernel_Cauchy = np.genfromtxt('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Cauchy_Kernel.txt', delimiter=',')
kernel_Cauchy = np.reshape(kernel_Cauchy,(3000,1))

kernel_Linear = kernel_Linear/R
kernel_Linear = 10*np.log10(kernel_Linear)

kernel_Polynomial = kernel_Polynomial/R
kernel_Polynomial = 10*np.log10(kernel_Polynomial)

kernel_Gaussiano = kernel_Gaussiano/R
kernel_Gaussiano = 10*np.log10(kernel_Gaussiano)

kernel_Exponential= kernel_Exponential/R
kernel_Exponential = 10*np.log10(kernel_Exponential)

kernel_Laplacian = kernel_Laplacian/R
kernel_Laplacian = 10*np.log10(kernel_Laplacian)

kernel_Sigmoid = kernel_Sigmoid/R
kernel_Sigmoid = 10*np.log10(kernel_Sigmoid)

kernel_Rational_Quadratic = kernel_Rational_Quadratic/R
kernel_Rational_Quadratic = 10*np.log10(kernel_Rational_Quadratic)

kernel_Multiquadric = kernel_Multiquadric/R
kernel_Multiquadric = 10*np.log10(kernel_Multiquadric)

kernel_Inverse_Multiquadric = kernel_Inverse_Multiquadric/R
kernel_Inverse_Multiquadric = 10*np.log10(kernel_Inverse_Multiquadric)

kernel_Cauchy = kernel_Cauchy/R
kernel_Cauchy = 10*np.log10(kernel_Cauchy)

plt.plot(kernel_Linear, color='blue')
plt.ylabel('Mean Squared Error (MSE) [dB] Linear')
plt.xlabel('iterations (n)')
plt.legend((''))
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Linear.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Polynomial, color='red')
plt.ylabel('Mean Squared Error (MSE) [dB] Polynomial')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Polynomial.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Gaussiano, color='green')
plt.ylabel('Mean Squared Error (MSE) [dB] Gaussian')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Gaussian.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Laplacian, color='orange')
plt.ylabel('Mean Squared Error (MSE) [dB] Laplacian')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Laplacian.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Inverse_Multiquadric, color='gold')
plt.ylabel('Mean Squared Error (MSE) [dB] Inverse Multiquadric')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Inverse Multiquadric.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Sigmoid, color='aqua')
plt.ylabel('Mean Squared Error (MSE) [dB] Sigmoid')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Sigmoid.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Exponential, color='royalblue')
plt.ylabel('Mean Squared Error (MSE) [dB] Exponential')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Exponential.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Rational_Quadratic, color='navy')
plt.ylabel('Mean Squared Error (MSE) [dB] Rational Quadratic')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Rational Quadratic.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Cauchy, color='purple')
plt.ylabel('Mean Squared Error (MSE) [dB] Cauchy')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Cauchy.eps', format='eps', dpi=300)
plt.figure()

plt.plot(kernel_Multiquadric, color='magenta')
plt.ylabel('Mean Squared Error (MSE) [dB] Multiquadric')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('/home/patrick/Desktop/Projeto de Pesquisa/Kernel_Library/Simulacao n=3k, r=50, mu = 30-4, sz=10-3, sx=10-2/Graphs/MSE/MSE Multiquadric.eps', format='eps', dpi=300)
plt.show()