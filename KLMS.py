import numpy as np
import matplotlib.pyplot as plt
import kernel_methods as kernel
import csv
import time

N = 100  # numero de iteracoes
R = 200  # numero de realizacoes
mu = 1 * 10 ** (-4)  # step-size

w0 = np.array(
    [[-0.0098], [-0.0143], [-0.0251], [-0.0399], [-0.0518], [-0.0619], [-0.0709], [-0.0781], [-0.0834], [-0.0869],
     [-0.0886], [-0.0886], [-0.0869], [-0.0836], [-0.0789], [-0.0729], [-0.0657], [-0.0576], [-0.0488], [-0.0394],
     [-0.0297], [-0.0198], [-0.0100], [-0.0004], [0.0088], [0.0175], [0.0254], [0.0325], [0.0387], [0.0439], [0.0480],
     [0.0510], [0.0530], [0.0538], [0.0536], [0.0524], [0.0503], [0.0473], [0.0435], [0.0391], [0.0341], [0.0287],
     [0.0230], [0.0172], [0.0112], [0.0053], [-0.0004], [-0.0059], [-0.0110], [-0.0157], [-0.0199], [-0.0236],
     [-0.0266], [-0.0290], [-0.0307], [-0.0318], [-0.0322], [-0.0320], [-0.0312], [-0.0299], [-0.0280], [-0.0257],
     [-0.0231], [-0.0201], [-0.0168], [-0.0134], [-0.0099], [-0.0064], [-0.0029], [0.0005], [0.0037], [0.0067],
     [0.0095], [0.0119], [0.0140], [0.0158], [0.0171], [0.0181], [0.0187], [0.0190], [0.0188], [0.0183], [0.0175],
     [0.0164], [0.0150], [0.0135], [0.0117], [0.0098], [0.0078], [0.0058], [0.0037], [0.0017], [-0.0003], [-0.0022],
     [-0.0039], [-0.0055], [-0.0069], [-0.0081], [-0.0091], [-0.0099], [-0.0105], [-0.0108], [-0.0109], [-0.0109],
     [-0.0106], [-0.0101], [-0.0095], [-0.0087], [-0.0078], [-0.0068], [-0.0057], [-0.0045], [0.0034], [-0.0022],
     [-0.0010], [0.0001], [0.0012], [0.0022], [0.0031], [0.0039], [0.0046], [0.0051], [0.0056], [0.0059], [0.0061],
     [0.0062], [0.0062], [0.0060], [0.0057], [0.0054], [0.0049], [0.0044], [0.0039], [0.0033], [0.0026], [0.0020],
     [0.0013], [0.0007], [0.0000], [-0.0006], [-0.0011], [-0.0016], [-0.0021], [-0.0025], [-0.0028], [-0.0031],
     [-0.0033], [-0.0034], [-0.0034], [-0.0034]])
# w0 = np.array([[1], [-2], [3]]) #just for test
# Diferente do codigo em matlab, aqui o w0 ja eh gerado como matriz coluna

sigma_z = 1 * 10 ** (-3)  # desvio padrao do ruido
sigma_x = 1.0 * 10 ** (-2)  # desvio padrão do sinal de entrada

# Algoritmo adaptativo
p = len(w0)
mse = np.zeros((N, 1))
Ew = np.zeros((p, N))
res = np.zeros((N))

sigma = 0.5  # para o kernel Gaussiano
degree = 1
aux = False
a = 30
c = 5

u = np.zeros((p, 1))
uk = np.zeros((p, 1))
dictionary = np.zeros((p, p))

aux_array = np.zeros(len(mse))


for r in range(1, R):
    e = np.zeros((N, 1))
    w = np.zeros((p, N))
    for i in range(0, p):
        for j in range(0, p):
            dictionary[i, j] = sigma_x * np.random.rand()
    if aux is False:
        for i in range(0, p):
            u[i] = sigma_x * np.random.rand()
        aux = True
    for i in range(0, N - 1):
        u = np.delete(u, p - 1, axis=0)
        aux3 = np.random.rand() * sigma_x
        u = np.insert(u, 0, aux3, axis=0)
        for v in range(0, p):
            uk[v] = kernel.Polynomial_K(dictionary[:, v], u[:], sigma, degree)
            #uk[v] = kernel.Gaussian_K(dictionary[:, v], u[:], sigma)
            #uk[v] = kernel.Exponential_K(dictionary[:, v], u[:], sigma)
            #uk[v] = kernel.Laplacian_K(dictionary[:, v], u[:], sigma)
            #uk[v] = kernel.Sigmoid_K(dictionary[:, v], u[:], a, c)
            #uk[v] = kernel.Rational_Quadratic_K(dictionary[:, v], u[:], c)
            #uk[v] = kernel.Inverse_Multiquadric_K(dictionary[:, v], u[:], c)
            #uk[v] = kernel.Multiquadric_K(dictionary[:, v], u[:], c)
            #uk[v] = kernel.Cauchy_K(dictionary[:, v], u[:], sigma)
            
        d = np.dot(w0.T, np.sin(u)) + np.sqrt(sigma_z) * np.random.rand()  # d esta saindo escalar
        e[i] = d - np.dot(w[:, i].T, uk)  # 'e' is shaped as (N, 1)
        w[:, i + 1] = w[:, i] + uk.T * e[i] * mu

    
"""
mse = e[:] ** 2 + mse
Ew = w[:, range(0, N)] + Ew  # w is shaped as (p, 1)
np.savetxt('put-linuxPath-here/Kernel_Library/Simulacao1/Polynomianl_Kernel.txt',mse, delimiter=',')

gaussian_K = np.genfromtxt('put-linuxPath-here/Kernel_Library/Simulacao1/Gaussian_Kernel.txt',delimiter=',')
cauchy_K = np.genfromtxt( 'put-linuxPath-here/Patrick/Kernel_Library/Simulacao1/Cauchy_Kernel.txt', delimiter=',')"""

#gaussian_K = np.reshape(gaussian_K, (len(mse), 1))
#cauchy_K = np.reshape(cauchy_K, (len(mse), 1))

#gaussian_K = gaussian_K / R
#cauchy_K = cauchy_K / R

MSE = mse / R
Ew = Ew / R


for i in range(1, N):
    res[i] = np.linalg.norm(Ew[:, i] - Ew[:, i - 1])
"""
# Gera os graficos
MSE = MSE.reshape(N)
plt.plot(res)
plt.xlabel('iterations(n)')
plt.ylabel('Relative error')
plt.grid(True)
plt.figure()

MSE = 10 * np.log10(MSE)
gaussian_K = 10 * np.log10(gaussian_K)
cauchy_K = 10 * np.log10(cauchy_K)

plt.plot(cauchy_K, color='green')
plt.plot(gaussian_K, color='red')
plt.plot(MSE, color='blue')
plt.legend(('Polinomial', 'Gaussiano', 'Cauchy'))
plt.xlim(0, 2000)
plt.ylabel('MSE [dB]')
plt.xlabel('Iterações (n)')
plt.grid(True)
plt.savefig(
    'C:/Users/patri/Desktop/Patrick/Kernel_Library/Simulacao1/Graphs/MSE/MSE '
    'Gaussiano (Vermelho) vs Cauchy(Verde) vs Polinomial(Azul).eps',
    format='eps', dpi=300)
plt.figure()

for i in range(len(w0)):
    plt.plot(Ew[i, :])
plt.ylabel('E/{/bf w}/}')
plt.xlabel('iterations (n)')
plt.grid(True)
plt.savefig('KLMS_Coeficientes_Linear.eps', format='eps', dpi=300)
plt.show()
"""


