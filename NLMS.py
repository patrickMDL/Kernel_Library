import numpy as np
import math 
import matplotlib.pyplot as plt
import random as rand


N = 2000 #numero de iteracoes

R = 10 #numero de realizacoes

mu = 1.0*10**(-1) #step-size

w0 =  np.array([[ 1],[-2], [-0.1], [0.2], [math.pi], [math.exp(-2.6)], [-math.sqrt(3)]])
#Diferente do codigo em matlab, aqui o w0 ja eh gerado como matriz coluna

sigma_z = 1*10**(-2) #desvio padrao do ruido
sigma_x = 1.0 #	desvio padrão do sinal de entrada

#Algoritmo adaptativo
mse = np.zeros((N,1))
Ew = np.zeros((len(w0), N))

for r in range(1,R):
	e = np.zeros((N, 1))
	x = np.zeros((len(w0),1))
	w = np.zeros((len(w0), N))

	v = 0
	for i in range(0, N-1):
		if (v+1>=len(w0)):
			x[v] = sigma_x * np.random.rand() 
		elif (v<len(w0)):
			x[v] = sigma_x * np.random.rand()
		
		d = np.dot( x.T, w0) + sigma_z*np.random.rand() #d esta saindo escalar
		e[i] = d - np.dot(w[:,i],x)  # 'e' is shaped as (2000, 1)

		
		if (v+1>=len(w0)):
			w[:,i+1] = w[:,i] + ((2*mu)/np.dot(x[:,0],x[:,0]))*e[i] * x[:,0] 
			v = 0
		elif (v<len(w0)):
			w[:,i+1] = w[:,i] + ((2*mu)/np.dot(x[:,0],x[:,0]))*e[i] * x[:,0]  
			v=v+1

	mse = e[:]**2 + mse
	Ew = w[:,range(0,N)] + Ew #w is shaped as (len(w0), 1)


MSE = mse/r
Ew = Ew/r


#Gera os graficos
MSE = MSE.reshape(N)
MSE=10*np.log10(MSE)
plt.plot(MSE)
plt.ylabel('Mean Squared Error (MSE) [dB] NLMS')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.figure()

for i in range (len(w0)):
	plt.plot ( Ew[i,:])
plt.ylabel('E\{{\bf w}\}NLMS')
plt.xlabel('iterations (n)')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()