import numpy as np

def porosity(im)->float:
    if im.any():
        shape=im.size 
        porosity=np.sum(im) / shape 
        return porosity
    return -1.0


x=np.array([[1,0],[1,1]])
print(porosity(x))
    