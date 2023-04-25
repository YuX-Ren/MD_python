import torch
from torch import nn

# Define the neural network model for the potential energy
class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

def calculate_forces(positions, energy_model):
    """
    Calculate forces using the energy model.
    positions: tensor of particle positions (shape: N x 3, N = number of particles)
    energy_model: the neural network model for the potential energy
    """
    positions.requires_grad = True
    N = positions.shape[0]
    forces = torch.zeros_like(positions)

    for i in range(N):
        energy = energy_model(positions[i])
        energy.backward(retain_graph=True)
        forces[i] = -positions.grad[i]
        positions.grad.data.zero_()

    return forces

def integrate_runge_kutta(positions, velocities, masses, forces, dt,energy_model):
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

    k21 = calculate_forces(positions + 0.5 * dt * k12,energy_model) / masses
    k22 = velocities + 0.5 * dt * k11

    k31 = calculate_forces(positions + 0.5 * dt * k22,energy_model) / masses
    k32 = velocities + 0.5 * dt * k21

    k41 = calculate_forces(positions + dt * k32,energy_model) / masses
    k42 = velocities +  dt * k31

    new_positions = positions + dt * (k12 + 2 * k22 + 2 * k32 + k42) / 6
    new_forces = calculate_forces(new_positions,energy_model)
    new_velocities = velocities + dt * (k11 + 2 * k21 + 2 * k31 + k41) / 6

    return new_positions, new_velocities, new_forces