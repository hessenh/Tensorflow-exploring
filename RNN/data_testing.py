import numpy as np



a = np.arange(100)
a = np.reshape(a, [10,10])
print a

b = a < 30
print b

print a[b]