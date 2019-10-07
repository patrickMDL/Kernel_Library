import numpy as np
a = np.zeros((5,1))
print (a)
for i in range (0, 10):
    a = np.delete(a,4, axis=0)
    a = np.insert(a,0,np.random.rand(), axis=0)
    print (a[0])