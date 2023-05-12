import numpy as np
import sys
import matplotlib.pyplot as plt
def projectile_motion(v,T,h):
    H = [0]*100
    v= int(v)
    H[0]=int(h)
    dT = int(T)/100
    for i in range(1,100):
        g = 9.8
        dv_dt = g
        v = v+dv_dt *dT
        H[i] = H[i-1] - v*dT
    return H

# if __name__ == '__main__':
v=1;T=100;h=10
H=projectile_motion(v,T,h)
plt.plot(H)
plt.show()