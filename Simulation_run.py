import torch
import MDAnalysis as mda
from MDAnalysis.coordinates import DCD
from motion_utils import EnergyModel,integrate_Beeman,calculate_forces
"""
unit : 10^-24g 10^-10m 10^-15s
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ke2 = 2.31*1e-4
Ar = 1.66 # 1u = 1.66 mass unit
# Simulation setup
n_steps = 10000
dt = 1

pdb_file = "ala15.pdb"
psf_file = "ala15.psf"

# Read the PDB and PSF structure
u = mda.Universe(psf_file, pdb_file)
masses = torch.tensor(u.atoms.masses*Ar, dtype=torch.float32).unsqueeze(dim=1).to(device)
trajectory_writer = DCD.DCDWriter("trajectory.dcd", u.atoms.n_atoms)
Energy_Protein = EnergyModel(u).to(device)
# Initialize positions, velocities, and forces
initial_positions = torch.tensor(u.atoms.positions, dtype=torch.float32).to(device)
initial_velocities = torch.zeros_like(initial_positions).to(device)
initial_forces = calculate_forces(initial_positions,Energy_Protein)
positions = initial_positions
velocities = initial_velocities
forces = initial_forces

# Run the simulation
for step in range(n_steps):
    positions, velocities, forces = integrate_Beeman(positions, velocities, masses, forces, dt, Energy_Protein)
    # Periodically print out the step number and total energy
    # if step % 1000 == 0:
    #     u.atoms.positions = positions.detach().cpu().numpy()
    #     trajectory_writer.write(u.atoms)
    #     total_energy = Energy_Protein(positions)
    #     print(f"Step {step}: Total Energy = {total_energy:.6f}")

# Save the final positions to a new PDB file
final_positions = positions.detach().cpu().numpy()
u.atoms.positions = final_positions
u.write("final_protein_structure.pdb")