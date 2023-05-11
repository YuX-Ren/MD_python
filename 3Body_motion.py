import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

def TBODY_potential(positions):
    N = positions.shape[0]
    V = np.zeros_like(positions)
    
    for i in range(N):
        for j in range(i+1, N):
            r_ij = positions[j] - positions[i]
            r = np.linalg.norm(r_ij)
            V += -1/r
    return V

def TBODY_gradient(positions):
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    ks=10
    for i in range(N):
        for j in range(i+1, N):
            r_ij = positions[j] - positions[i]
            r = np.linalg.norm(r_ij)
            # dV_dr = -r
            dV_dr = -1/r**2
            force = -ks*dV_dr * r_ij / r
            forces[i] += force
            forces[j] -= force
    return forces

def integrate_Beeman(positions, velocities, masses, forces, dt):
    new_positions = positions + dt * velocities + dt * dt * 2/3 * TBODY_gradient(positions)/masses-1/6* dt *dt *forces/masses
    new_velocities = velocities + dt * 1/6 * (2 * TBODY_gradient(new_positions)/masses + 5 * TBODY_gradient(positions)/masses-forces/masses)
    return new_positions, new_velocities, TBODY_gradient(positions)

def integrate_runge_kutta(positions, velocities, masses, forces, dt):
    """
    Integrate the equations of motion using the 4th-order Runge-Kutta method.
    positions: tensor of particle positions (shape: N x 3, N = number of particles)
    velocities: tensor of particle velocities (shape: N x 3)
    masses: tensor of particle masses (shape: N)
    forces: tensor of forces acting on each particle (shape: N x 3)
    dt: time step for the integration
    energy_model: the neural network model for the potential energy
    """
    k11 = forces / masses
    k12 = velocities

    k21 = TBODY_gradient(positions + 0.5 * dt * k12) / masses
    k22 = velocities + 0.5 * dt * k11

    k31 = TBODY_gradient(positions + 0.5 * dt * k22) / masses
    k32 = velocities + 0.5 * dt * k21

    k41 = TBODY_gradient(positions + dt * k32) / masses
    k42 = velocities +  dt * k31

    new_positions = positions + dt * (k12 + 2 * k22 + 2 * k32 + k42) / 6
    new_forces = TBODY_gradient(new_positions)
    new_velocities = velocities + dt * (k11 + 2 * k21 + 2 * k31 + k41) / 6

    return new_positions, new_velocities, new_forces


# Simulation parameters
N = 3 # number of particles
dt = 0.01
n_steps = 1000


# Initialize positions, velocities, and masses

positions = np.random.rand(N, 3)*10
positions = positions-positions.sum(axis=0)/N
velocities = np.random.rand(N, 3)
velocities = velocities-velocities.sum(axis=0)/N


# positions = np.array([[2,1],[-1,1],[1,-1]],dtype=np.float64)
# positions = np.random.rand(N, 2)*10
# positions = positions-positions.sum(axis=0)/N
# velocities = np.random.rand(N, 2)*10
# velocities = velocities-velocities.sum(axis=0)/N
# print(velocities.sum(axis=0))
# velocities = np.zeros((N,2))


masses = np.ones(N)
masses=np.expand_dims(masses, axis=1)
forces = TBODY_gradient(positions)

r=np.zeros((n_steps+1, N, 3))
r[0]=positions
for step in range(n_steps):
    positions, velocities, forces = integrate_Beeman(positions, velocities, masses, forces, dt)
    r[step+1]=positions





# # Create a 3D scatter plot

# fig, ax = plt.subplots(figsize=(6,6))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter([], [], [])

# scatter = ax.scatter(r[0,:,0], r[0,:,1], marker='.', s=1000)

# Set plot limits (adjust these based on the expected range of positions)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)


ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Update function for the animation
def update(frame):
    # scatter.set_offsets(r[frame]) 


    positions = r[frame]
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    return scatter,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(r), interval=0.01, blit=False,)

# Display the animation
plt.show()
# Run the simulation
# ani.save('animation.gif')
# plt.close()
# Save the animation as a video file

