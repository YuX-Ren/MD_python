import torch
from torch import nn
import MDAnalysis as mda
from load_coe import parse_forcefield_xml
import numpy as np
# Define the neural network model for the potential energy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
unit : 10^-24g 10^-10m 10^-15s
"""
ke2 = 2.31*1e-4
Ar = 1.66 # 1u = 1.66 mass unit
kcal = 6.9501*1e-5

atom_type_mapping = {"CA": 0,"CB":1,"C": 2, "N": 3, "O":4, "OXT":5,"H":6,"H2":6,"H3":6,"HB3":6,"HB2":6,"HB1":6,"HA":6}  # Add other atom types if necessary
lj_atom_type_mapping = {"CA": 0,"CB":0,"C": 0, "N": 1, "O":2, "OXT":2,"H":3,"H2":3,"H3":3,"HB3":3,"HB2":3,"HB1":3,"HA":3}


n_atom_types = len(atom_type_mapping)
lj_n_atom_types = len(lj_atom_type_mapping)

lj_E = np.zeros((lj_n_atom_types, lj_n_atom_types))
lj_r = np.zeros((lj_n_atom_types, lj_n_atom_types))

lj_E[0,0] = 0.12*kcal;lj_E[0,1] = 0.155*kcal;lj_E[0,2] = 0.166*kcal;lj_E[0,3] = 0.11*kcal
lj_E[1,1] = 0.20*kcal;lj_E[1,2] = 0.214*kcal;lj_E[1,3] = 0.155*kcal
lj_E[2,2] = 0.23*kcal;lj_E[2,3] = 0.166*kcal
lj_E[3,3] = 0.12*kcal
lj_r[0,0] = 3.0;lj_r[0,1] = 2.85;lj_r[0,2] = 2.85;lj_r[0,3] = 2.4
lj_r[1,1] = 2.7;lj_r[1,2] = 2.7;lj_r[1,3] = 2.35
lj_r[2,2] = 2.7;lj_r[2,3] = 2.35
lj_r[3,3] = 2.0

lj_E_T = torch.tensor(lj_E, requires_grad=True)
lj_r_T = torch.tensor(lj_r, requires_grad=True)

file_path = 'FF.xml'
bond_coeffs,bond_lengths, angle_coeffs, thetas,dihedral_coeffs,phis,multiplicity = parse_forcefield_xml(file_path)
bond_coeffs = torch.tensor(bond_coeffs, requires_grad=True).to(device)
bond_lengths = torch.tensor(bond_lengths, requires_grad=True).to(device)
angle_coeffs = torch.tensor(angle_coeffs, requires_grad=True).to(device)
thetas = torch.tensor(thetas, requires_grad=True).to(device)
dihedral_coeffs = torch.tensor(dihedral_coeffs, requires_grad=True).to(device)
phis = torch.tensor(phis, requires_grad=True).to(device)
multiplicity = torch.tensor(multiplicity, requires_grad=True).to(device)
# pdb_file = "your_protein_structure.pdb"
# psf_file = "your_protein_structure.psf"

# # Read the PDB and PSF structure
# u = mda.Universe(psf_file, pdb_file)
# partial_model = u.select_atoms('not (name H and bonded name C)')

class EnergyModel(nn.Module):
    def __init__(self, universe):
        super(EnergyModel, self).__init__()
        self.atoms,  self.bonds, self.angles, self.dihedrals = self.get_topology_data(universe)

    def get_topology_data(self, universe):
        atoms_list = list(universe.atoms)
        # print(atoms_list)
        bonds = list(universe.bonds)
        # for bond in bonds:
        #     atom1, atom2= bond
        #     print(str(atom1.index)+' '+str(atom2.index))
        #     print(atom1.name+atom2.name)
        angles = list(universe.angles)
        dihedrals = list(universe.dihedrals)

        return atoms_list,  bonds, angles, dihedrals

    def get_distance_map(self, positions):
        distances_map = torch.cdist(positions, positions)
        return distances_map

    def bond(self, positions, atoms_list, bonds):
        bond_energy = 0
        for bond in bonds:
            atom1, atom2 = bond
            bond_distance = torch.norm(positions[atom1.index] - positions[atom2.index])
            # Use bond_coeffs and bond_lengths for your specific system
            k_bond = bond_coeffs[atom_type_mapping[atom1.name], atom_type_mapping[atom2.name]]
            r0 = bond_lengths[atom_type_mapping[atom1.name], atom_type_mapping[atom2.name]]
            bond_energy += 0.5 * k_bond * (bond_distance - r0)**2
        return bond_energy

    def angle(self, positions, atoms_list, angles):
        angle_energy = 0
        for angle_triplet in angles:
            atom1, atom2, atom3 = angle_triplet
            vec1 = positions[atom1.index] - positions[atom2.index]
            vec2 = positions[atom3.index] - positions[atom2.index]
            cosine_angle = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
            angle_rad = torch.acos(cosine_angle)
            # Use angle_coeffs and angles for your specific system
            k_angle = angle_coeffs[atom_type_mapping[atom1.name], atom_type_mapping[atom2.name], atom_type_mapping[atom3.name]]
            theta0 = thetas[atom_type_mapping[atom1.name], atom_type_mapping[atom2.name], atom_type_mapping[atom3.name]]
            # theta0 = angles[atom_type_mapping[atom1.name], atom_type_mapping[atom2.name], atom_type_mapping[atom3.name]]
            # theta0 = 109.8
            angle_energy += 0.5 * k_angle * (angle_rad - theta0)**2
        return angle_energy

    def dihedral(self, positions, atoms_list, dihedrals):
        dihedral_energy = 0
        for dihedral_quadruplet in dihedrals:
            atom1, atom2, atom3, atom4 = dihedral_quadruplet
            vec1 = positions[atom2.index] - positions[atom1.index]
            vec2 = positions[atom3.index] - positions[atom2.index]
            vec3 = positions[atom4.index] - positions[atom3.index]
            normal1 = torch.cross(vec1, vec2)
            normal2 = torch.cross(vec2, vec3)
            cosine_dihedral = torch.dot(normal1, normal2) / (torch.norm(normal1) * torch.norm(normal2))
            sine_dihedral = torch.dot(torch.cross(normal1, normal2), vec2) / (torch.norm(vec2) * torch.norm(normal1) * torch.norm(normal2))
            dihedral_rad = torch.atan2(sine_dihedral, cosine_dihedral)
            
            # Use dihedral_coeffs and dihedrals for your specific system
            k_dihedral = dihedral_coeffs[atom_type_mapping[atom2.name], atom_type_mapping[atom3.name]]
            delta_phase = phis[atom_type_mapping[atom2.name], atom_type_mapping[atom3.name]]
            n = multiplicity[atom_type_mapping[atom2.name], atom_type_mapping[atom3.name]]
            # dihedral_energy += 0.5 * k_dihedral[0] * (1 + torch.cos( dihedral_rad - delta_phase))+0.5 * k_dihedral[1] * (1 + torch.cos( dihedral_rad *2- delta_phase))+0.5 * k_dihedral[2] * (1 + torch.cos( 3*dihedral_rad - delta_phase))
            dihedral_energy += 0.5 * k_dihedral* (1 + torch.cos( n*dihedral_rad - delta_phase))
        return dihedral_energy

    def Lennard_Jones_potential(self, distances_map, atoms_list):
        lj_energy = 0
        n_atoms = len(atoms_list)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                atom1 = atoms_list[i]
                atom2 = atoms_list[j]
                r = distances_map[i, j]
                
                # Use lj_coeffs for your specific system
                epsilon = lj_E_T[atom_type_mapping[atom1.name], atom_type_mapping[atom2.name]]
                sigma = lj_r_T[atom_type_mapping[atom1.name], atom_type_mapping[atom2.name]]
                r6 = (sigma / r) ** 6
                r12 = r6 * r6
                lj_energy += 4 * epsilon * (r12 - r6)
        return lj_energy

    def coulomb(self, distances_map, atoms_list):
        coulomb_energy = 0
        n_atoms = len(atoms_list)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                atom1 = atoms_list[i]
                atom2 = atoms_list[j]
                r = distances_map[i, j]
                
                # Use charges from your specific system
                charge1 = atom1.charge
                charge2 = atom2.charge
                coulomb_energy += ke2*(charge1 * charge2) / r
        return coulomb_energy

    def forward(self, positions):
        distances_map = self.get_distance_map(positions)
        bond_energy = self.bond(positions, self.atoms, self.bonds)
        angle_energy = self.angle(positions, self.atoms, self.angles)
        dihedral_energy = self.dihedral(positions, self.atoms, self.dihedrals)
        lj_energy = self.Lennard_Jones_potential(distances_map, self.atoms)
        coulomb_energy = self.coulomb(distances_map, self.atoms)
        
        total_energy = bond_energy + angle_energy + dihedral_energy + lj_energy + coulomb_energy
        return total_energy

def calculate_forces(positions, energy_model):
    """
    Calculate forces using the energy model.
    positions: tensor of particle positions (shape: N x 3, N = number of particles)
    energy_model: the neural network model for the potential energy
    """
    F_positions = positions.clone()
    F_positions.requires_grad = True
    N = positions.shape[0]
    forces = torch.zeros_like(positions)

    energy = energy_model(F_positions)
    print(energy)
    energy.backward(retain_graph=True)
    forces = -F_positions.grad

    return forces

def integrate_Beeman(positions, velocities, masses, forces, dt,energy_model):
    cur_forces = calculate_forces(positions,energy_model)
    new_positions = positions + dt * velocities + dt * dt * 2/3 * cur_forces/masses-1/6* dt *dt *forces/masses
    new_velocities = velocities + dt * 1/6 * (2 * calculate_forces(new_positions,energy_model)/masses + 5 * cur_forces/masses-forces/masses)
    return new_positions, new_velocities, cur_forces