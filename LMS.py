import numpy as np
import math 
import matplotlib.pyplot as plt
import random as rand

N = 10000 #numero de iteracoes
	
R = 10 #numero de realizacoes

mu = 0.0001*10**(-1) #step-size
w0 = np.array([[-0.0098],   [-0.0143],   [-0.0251],   [-0.0399],   [-0.0518],   [-0.0619],   [-0.0709],  [-0.0781],   [-0.0834],   [-0.0869],   [-0.0886],   [-0.0886],   [-0.0869],   [-0.0836], [-0.0789],   [-0.0729],   [-0.0657],   [-0.0576],   [-0.0488],   [-0.0394],   [-0.0297], [-0.0198],  [-0.0100],   [-0.0004],    [0.0088],    [0.0175],    [0.0254],    [0.0325], [0.0387],   [0.0439],    [0.0480],    [0.0510],    [0.0530],    [0.0538],    [0.0536], [0.0524],   [0.0503],   [0.0473],    [0.0435],    [0.0391],    [0.0341],    [0.0287], [0.0230],   [0.0172],   [ 0.0112],    [0.0053],   [-0.0004],  [-0.0059],   [-0.0110], [-0.0157],  [-0.0199],   [-0.0236],   [-0.0266],  [-0.0290],   [-0.0307],   [-0.0318], [-0.0322],  [-0.0320],  [-0.0312],   [-0.0299],   [-0.0280],   [-0.0257],   [-0.0231], [-0.0201],  [-0.0168],   [-0.0134],   [-0.0099],   [-0.0064],   [-0.0029],    [0.0005], [0.0037],   [0.0067],   [ 0.0095],    [0.0119],    [0.0140],    [0.0158],    [0.0171], [0.0181],   [0.0187],   [ 0.0190],    [0.0188],   [0.0183],    [0.0175 ],   [0.0164], [0.0150],   [0.0135],   [ 0.0117],   [ 0.0098],   [ 0.0078],   [0.0058],    [0.0037], [0.0017],   [-0.0003], [-0.0022],   [-0.0039],   [-0.0055],   [-0.0069],   [-0.0081], [-0.0091],  [-0.0099],  [-0.0105],   [-0.0108],   [-0.0109],   [-0.0109],   [-0.0106], [-0.0101],  [-0.0095], [-0.0087],   [-0.0078],   [-0.0068],   [-0.0057],   [-0.0045], [0.0034],  [-0.0022],  [-0.0010],   [ 0.0001],    [0.0012],   [0.0022],    [0.0031], [0.0039],   [0.0046],   [0.0051],   [ 0.0056],   [0.0059],    [0.0061],    [0.0062], [0.0062],   [0.0060],   [0.0057],   [ 0.0054],    [0.0049],    [0.0044],   [0.0039],[0.0033],   [0.0026],   [0.0020],    [0.0013],    [0.0007],    [0.0000],   [-0.0006], [-0.0011],  [-0.0016], [-0.0021],   [-0.0025],   [-0.0028],  [-0.0031],  [-0.0033], [-0.0034],  [-0.0034],  [-0.0034]])
#w0 =  np.array([[ 1],[-2], [-0.1], [0.2], [math.pi], [math.exp(-2.6)], [-math.sqrt(3)]])
#Diferente do codigo em matlab, aqui o w0 ja eh gerado como matriz coluna

#Estatistica do sinal de entrada
sigma_z = 1*10**(-3) #desvio padrao do ruido
sigma_x = 1.0 #desvio padrao do sinal de entrada

#algoritmo adaptativo
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
			w[:,i+1] = w[:,i] + e[i] * x[:,0] * mu
			v = 0
		elif (v<len(w0)):
			w[:,i+1] = w[:,i] + e[i] * x[:,0] * mu
			v=v+1

	mse = e[:]**2 + mse
	Ew = w[:,range(0,N)] + Ew #w is shaped as (len(w0), 1)

MSE = mse/r
Ew = Ew/r

#Gerar graficos

MSE = MSE.reshape(N)
MSE=10*np.log10(MSE)
plt.plot(MSE)
plt.ylabel('Mean Squared Error (MSE) [dB] LMS')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.figure()

for i in range (len(w0)):
	plt.plot ( Ew[i,:])
plt.ylabel('E\{{\bf w}\} LMS')
plt.xlabel('iterations (n)')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()