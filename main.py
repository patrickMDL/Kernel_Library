import kernel_methods as km
import numpy as np

x = np.array ([1,2,3,4])
y = np.array ([4,3,2,1])
a = 0.0003
s = 5
c = 4
d = 0.0004
print ("\033[1;34;40m================================================================================")
print("\t\t\t\t Kernel Functions  \n")
print ("Used Values\n**x=%s" % x) 
print ("**y = %s" % y) 
print ("**a(alfa) = %s" % a)
print ("**s(sigma) = %s" % s)
print ("**c(constant) = %s" % c)
print ("**d(degree) = %s" % d)

print ("-------------------------------------------------------------------------------")

result_Linear = km.Linear_K(x,y,c)
print ("Linear Kernel: %s" % result_Linear)

result_Polynomial = km.Polynomial_K( x, y, a, c, d)
print ("Polynomial Kernel: %s" % result_Polynomial)

result_Gaussian = km.Gaussian_K( x, y, s)
print ("Gaussian Kernel: %s" % result_Gaussian)

result_Exponential = km.Exponential_K(x, y, s)
print ("Exponential Kernel: %s" % result_Exponential)

result_Laplacian = km.Laplacian_K(x, y, s)
print ("Laplacian Kernel: %s" % result_Laplacian)

result_Sigmoid = km.Sigmoid_K(x, y, a, c)
print ("Sigmoid Kernel: %s" % result_Sigmoid)

result_Rational = km.Rational_Quadratic_K(x, y, c)
print ("Rational Kernel: %s" % result_Rational)

result_Multiquadric = km.Multiquadric_K (x, y, c)
print ("Multiquadric Kernel: %s" % result_Multiquadric)

result_Inverse = km.Inverse_Multiquadric_K(x,y,c)
print ("Inverse Multiquadric Kernel: %s" % result_Inverse)

result_Cauchy = km.Cauchy_K(x, y, s)
print ("Cauchy Kernel: %s" % result_Cauchy)

print ("================================================================================")