import torch
import MDAnalysis as mda
from MDAnalysis.coordinates import DCD
from GIN.GIN import integrate_Beeman,calculate_forces
"""
unit : 10^-24g 10^-10m 10^-15s
"""
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Ar = 1.66 # 1u = 1.66 mass unit
# Simulation setup
n_steps = 100
dt = 0.1

pdb_file = "ala15.pdb"
psf_file = "ala15.psf"

# Read the PDB and PSF structure
u = mda.Universe(psf_file, pdb_file)
masses = torch.tensor(u.atoms.masses*Ar, dtype=torch.float32).unsqueeze(dim=1).to(device)
trajectory_writer = DCD.DCDWriter("test.dcd", u.atoms.n_atoms)
forces_func = calculate_forces(u,2,32).to(device)
# Initialize positions, velocities, and forces
initial_positions = torch.tensor(u.atoms.positions, dtype=torch.float32).to(device)
initial_velocities = torch.zeros_like(initial_positions).to(device)*0.01
initial_forces = forces_func(initial_positions)
positions = initial_positions
velocities = initial_velocities
forces = cur_forces = initial_forces

# Run the simulation
for step in range(n_steps):
    positions, velocities, forces, cur_forces = integrate_Beeman(positions, velocities, masses, forces, cur_forces, dt, forces_func)
    # Periodically print out the step number and total energy
    # if step % 1000 == 0:
    u.atoms.positions = positions.detach().cpu().numpy()
    trajectory_writer.write(u.atoms)

# Save the final positions to a new PDB file
final_positions = positions.detach().cpu().numpy()
# u.atoms.positions = final_positions
# u.write("final_protein_structure.pdb")