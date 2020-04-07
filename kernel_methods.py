import numpy as np
e = 2.718281828459045235360287
def Linear_K( x, y, a):
   #x and y are the vector, while c is the constant
   x = np.reshape(x,(1,len(x)))
   y = np.reshape(y,(1,len(y)))
   result = (np.dot(x,y.T) + a)
   return result

def Polynomial_K(x, y, a,d): #note: A polynomial with d = 1, is equivalent to the linear kernel;
    #a stands for alfa and 'd' the degree
    x = np.reshape(x,(1,len(x)))
    y = np.reshape(y,(1,len(y)))
    result = (np.dot(x,y.T) + a)**d
    return result

def Gaussian_K (x, y, s):
   #S stands for sigma 
   norm = np.linalg.norm(x-y)
   div = 2*(s**2)
   exponencial = (norm**2 / div) * -1
   result = np.exp(exponencial)
   return result

def Exponential_K(x, y, s):
   result = np.exp(-1*(np.linalg.norm(x-y)/(2*(s**2))))
   return result

def Laplacian_K(x, y, s):
   result = np.exp(-1*(np.linalg.norm(x-y)/s))
   return result

def Sigmoid_K(x, y, a, c):
   result = np.tanh(a*np.dot(x.T,y)+c)
   return result

def Rational_Quadratic_K(x, y, c):
   result = 1 - ((np.linalg.norm(x-y)**2)/((np.linalg.norm(x-y)**2)+c))
   return result

def Multiquadric_K (x, y, c):
   result = np.sqrt(np.linalg.norm(x-y)**2 + c**2)
   return result

def Inverse_Multiquadric_K(x,y,c):
   result = 1/np.sqrt(np.linalg.norm(x-y)**2 + c**2)
   return result

def Cauchy_K(x, y, s):
   result = (np.linalg.norm(x-y)**2/(s**2))
   result = 1 + result
   result = 1 / result
   return result
