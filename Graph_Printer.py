import numpy as np
import math 
import matplotlib.pyplot as plt
import random as rand
import kernel_methods as kernel 
import csv


kernel_Linear = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Linear = np.reshape(result,(len(mse),1))


kernel_Polynomial = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Polynomial = np.reshape(result,(len(mse),1))


kernel_Gaussiano = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Gaussiano = np.reshape(result,(len(mse),1))

kernel_Exponential = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Exponential = np.reshape(result,(len(mse),1))

kernel_Laplacian = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Laplacian = np.reshape(result,(len(mse),1))

kernel_Sigmoid = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Sigmoid = np.reshape(result,(len(mse),1))

kernel_Rational_Quadratic = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Rational_Quadratic = np.reshape(result,(len(mse),1))

kernel_Multiquadric = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Multiquadric = np.reshape(result,(len(mse),1))

kernel_Inverse_Multiquadric = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Inverse_Multiquadric = np.reshape(result,(len(mse),1))

kernel_Cauchy = np.genfromtxt('KLMSteste.txt', delimiter=',')
kernel_Cauchy = np.reshape(result,(len(mse),1))


RESULT = result/R
RESULT=10*np.log10(RESULT)


plt.plot(MSE, color='blue')
plt.plot(RESULT, color='red')
plt.legend('EG')
plt.ylabel('Mean Squared Error (MSE) [dB]')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('MSE Gaussiano(red) vs Exponential(blue).eps', format='eps', dpi=300)
plt.figure()