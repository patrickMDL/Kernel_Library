import numpy as np
e = 2.718281828

def Linear_K( x, y, c):
    #x and y are the vector, while c is the constant
    #k(x,y)=x^T.y + c
   result = y.dot(x.T) + c
   return result

def Polynomial_K(x, y, a, c, d):
    #a stands for alfa and 'd' the degree
    result = (a*(y.dot(x.T)) + c)**d
    return result

def Gaussian_K (x, y, s):
   #S stands for sigma and e is the euler number
    x=x-y
    result = e**(-1*(np.linalg.norm(x)**2/2*(s**2)))
    return result

def Exponential_K(x, y, s):
   x = x-y
   result = e**(-1*(np.linalg.norm(x)/2*(s**2)))
   return result

def Laplacian_K(x, y, s):
   x = x-y
   result = e**(-1*(np.linalg.norm(x)/s))
   return result

def Sigmoid_K(x, y, a, c):
   result = (a*y.dot(x.T))+c
   result = np.sinh(result)/np.cosh(result)
   return result

def Rational_Quadratic_K(x, y, c):
   x=x-y
   result = 1 - ((np.linalg.norm(x)**2)/((np.linalg.norm(x)**2)+c))
   return result

def Multiquadric_K (x, y, c):
   norm = x-y
   norm = np.linalg.norm(norm)**2 + c**2
   result = np.sqrt(norm)
   return result

def Inverse_Multiquadric_K(x,y,c):
   norm = x-y
   norm = np.linalg.norm(norm)**2 + c**2
   result = 1/np.sqrt(norm)
   return result

def Cauchy_K(x, y, s):
   x=x-y
   result = (np.linalg.norm(x)**2/(s**2))
   result = 1 + result
   result = 1 / result
   return result
