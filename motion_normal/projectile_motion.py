import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

def projectile_motion(v, T, h):
    r=np.zeros((1000, 2))
    V = 0
    r[0][1] = int(h)
    r[0][0] = 0
    dT = int(T) / 1000

    for i in range(1, 1000):
        g = 9.8
        dv_dt = g
        V = V + dv_dt * dT
        r[i][1] = r[i-1][1] - V * dT
        r[i][0] = r[i-1][0] + v * dT

    return r

# Set up initial parameters
v = 1
T = 10
h = 500

# Calculate projectile motion
r = projectile_motion(v, T, h)

fig, ax = plt.subplots(figsize=(6,6))

scatter = ax.scatter(r[0,0], r[0,1], marker='.', s=1000)
# Set plot limits (adjust these based on the expected range of positions)
ax.set_xlim(0, 10)
ax.set_ylim(0, 520)

# Update function for the animation
def update(frame):
    scatter.set_offsets(r[frame]) 
    return scatter,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(r), interval=0.01, blit=False,)
from matplotlib.animation import PillowWriter
ani.save('h.gif', writer=PillowWriter(fps=60))
# Display the animation
plt.close()