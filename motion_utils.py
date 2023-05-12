import torch
from torch import nn
import MDAnalysis as mda
from load_coe import parse_forcefield_xml
import numpy as np
# Define the neural network model for the potential energy
# device = torch.device('cpu')
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
lj_r[0,0] = 3.0;lj_r[0,1] = 2.85;lj_r[0,2] = 2.85;lj_r[0,3] = 0
lj_r[1,1] = 2.7;lj_r[1,2] = 2.7;lj_r[1,3] = 2.35
lj_r[2,2] = 2.7;lj_r[2,3] = 2.35
lj_r[3,3] = 0

lj_E_T = torch.tensor(lj_E, requires_grad=False).to(device)
lj_r_T = torch.tensor(lj_r, requires_grad=False).to(device)

file_path = 'FF.xml'
bond_coeffs,bond_lengths, angle_coeffs, thetas,dihedral_coeffs,phis,multiplicity = parse_forcefield_xml(file_path)
bond_coeffs = torch.tensor(bond_coeffs, requires_grad=False).to(device)
bond_lengths = torch.tensor(bond_lengths, requires_grad=False).to(device)
angle_coeffs = torch.tensor(angle_coeffs, requires_grad=False).to(device)
thetas = torch.tensor(thetas, requires_grad=False).to(device)
dihedral_coeffs = torch.tensor(dihedral_coeffs, requires_grad=False).to(device)
phis = torch.tensor(phis, requires_grad=False).to(device)
multiplicity = torch.tensor(multiplicity, requires_grad=False).to(device)
# pdb_file = "your_protein_structure.pdb"
# psf_file = "your_protein_structure.psf"

# # Read the PDB and PSF structure
# u = mda.Universe(psf_file, pdb_file)
# partial_model = u.select_atoms('not (name H and bonded name C)')

class EnergyModel(nn.Module):
    def __init__(self, universe):
        super(EnergyModel, self).__init__()
        self.atoms,  self.bonds, self.angles, self.dihedrals, self.charges = self.get_topology_data(universe)
        n_atoms = len(self.atoms)
        self.epsilon = torch.zeros((n_atoms,n_atoms),dtype=float).to(device)
        self.sigma = torch.zeros((n_atoms,n_atoms),dtype=float).to(device)
        self.charges_pair = torch.zeros((n_atoms,n_atoms),dtype=float).to(device)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if(self.atoms[i]>self.atoms[j]):# i>j [i,j]=0
                    self.epsilon[i,j] = lj_E_T[self.atoms[j], self.atoms[i]]
                    self.sigma[i,j] = lj_r_T[self.atoms[j], self.atoms[i]]
                # Use lj_coeffs for your specific system
                self.epsilon[i,j] = lj_E_T[self.atoms[i], self.atoms[j]]
                self.sigma[i,j] = lj_r_T[self.atoms[i], self.atoms[j]]
                self.charges_pair[i,j] = self.charges[i]*self.charges[j]
        self.k_bond = torch.zeros(len(self.bonds),dtype=float).to(device)
        self.r0 = torch.zeros(len(self.bonds),dtype=float).to(device)
        for i,bond in enumerate(self.bonds):
            (atom1_type, atom1_index), (atom2_type, atom2_index) = bond
            # Use bond_coeffs and bond_lengths for your specific system
            if(atom1_index>atom2_index):
                self.charges_pair[atom2_index,atom1_index] = 0
                # self.epsilon[atom2_index,atom1_index] = 0
            if(bond_coeffs[atom2_type,atom1_type]!=0):
                self.k_bond[i] = bond_coeffs[atom2_type,atom1_type]
                self.r0[i] = bond_lengths[atom2_type,atom1_type]
            elif(bond_coeffs[atom1_type,atom2_type]!=0):
                self.k_bond[i] = bond_coeffs[atom1_type,atom2_type]
                self.r0[i] = bond_lengths[atom1_type,atom2_type]
            else:
                print(atom1_type,atom2_type)
                raise("value_error")
            self.charges_pair[atom1_index,atom2_index] = 0
            # self.epsilon[atom1_index,atom2_index] = 0
        
        self.k_angle = torch.zeros(len(self.angles),dtype=float).to(device)
        self.theta0 = torch.zeros(len(self.angles),dtype=float).to(device)
        for i,angle_triplet in enumerate(self.angles):
            (atom1_type, atom1_index), (atom2_type, atom2_index), (atom3_type, atom3_index) = angle_triplet
            if(angle_coeffs[atom3_type,atom2_type,atom1_type]!=0):
                self.k_angle[i] = angle_coeffs[atom3_type,atom2_type,atom1_type]
                self.theta0[i] = thetas[atom3_type,atom2_type,atom1_type]
            elif(angle_coeffs[atom1_type,atom2_type,atom3_type]!=0):
                self.k_angle[i] = angle_coeffs[atom1_type,atom2_type,atom3_type]
                self.theta0[i] = thetas[atom1_type,atom2_type,atom3_type]
            else:
                print(atom1_type,atom2_type,atom3_type)
                raise("value_error")
        self.k_dihedral = torch.zeros(len(self.dihedrals),dtype=float).to(device)
        self.default_phase = torch.zeros(len(self.dihedrals),dtype=float).to(device)
        self.n = torch.zeros(len(self.dihedrals),dtype=float).to(device)
        for i,dihedral_quadruplet in enumerate(self.dihedrals):
            (atom1_type, atom1_index), (atom2_type, atom2_index), (atom3_type, atom3_index), (atom4_type, atom4_index) = dihedral_quadruplet
            if(dihedral_coeffs[atom3_type,atom2_type]!=0):
                self.k_dihedral[i] = dihedral_coeffs[atom3_type,atom2_type]
                self.default_phase[i] = phis[atom3_type,atom2_type]
                self.n[i] = multiplicity[atom3_type,atom2_type]
            elif(dihedral_coeffs[atom2_type,atom3_type]!=0):
                self.k_dihedral[i] = dihedral_coeffs[atom2_type,atom3_type]
                self.default_phase[i] = phis[atom2_type,atom3_type]
                self.n[i] = multiplicity[atom2_type,atom3_type]
            else:
                raise("value_error")
    def get_topology_data(self, universe):
        atoms_list = torch.tensor([lj_atom_type_mapping[atom.name] for atom in universe.atoms])
        bonds = [((atom_type_mapping[bond[0].name], bond[0].index), (atom_type_mapping[bond[1].name], bond[1].index)) for bond in universe.bonds]

        angles = [((atom_type_mapping[angle[0].name], angle[0].index), (atom_type_mapping[angle[1].name], angle[1].index), (atom_type_mapping[angle[2].name], angle[2].index)) for angle in universe.angles]

        dihedrals = [((atom_type_mapping[dihedral[0].name], dihedral[0].index), (atom_type_mapping[dihedral[1].name], dihedral[1].index), (atom_type_mapping[dihedral[2].name], dihedral[2].index), (atom_type_mapping[dihedral[3].name], dihedral[3].index)) for dihedral in universe.dihedrals]

        charges = torch.tensor([atom.charge for atom in universe.atoms]).unsqueeze(dim=1).to(device)
        # bonds = list(universe.bonds)
        # # for bond in bonds:
        # #     atom1, atom2= bond
        # #     print(str(atom1.index)+' '+str(atom2.index))
        # #     print(atom1.name+atom2.name)
        # angles = list(universe.angles)
        # dihedrals = list(universe.dihedrals)

        return atoms_list,  bonds, angles, dihedrals, charges

    def get_distance_map(self, positions):
        distances_map = torch.cdist(positions, positions).to(device)+0.000000001
        return distances_map

    def bond(self, distances_map, bonds):
        bond_energy = 0
        bond_distances = torch.zeros(len(bonds),dtype=float).to(device)
        for i,bond in enumerate(bonds):
            (atom1_type, atom1_index), (atom2_type, atom2_index) = bond
            bond_distances[i] = distances_map[atom1_index,atom2_index]
            # Use bond_coeffs and bond_lengths for your specific system
        bond_energy = (0.5 * self.k_bond * (bond_distances - self.r0)**2).sum()
        print(((bond_distances - self.r0)**2).sum())
        return bond_energy

    def angle(self, positions,  angles):
        angle_energy = 0
        vec1 = torch.zeros((len(angles),3),dtype=float).to(device)
        vec2 = torch.zeros((len(angles),3),dtype=float).to(device)
        for i,angle_triplet in enumerate(angles):
            (atom1_type, atom1_index), (atom2_type, atom2_index), (atom3_type, atom3_index) = angle_triplet
            vec1[i] = positions[atom2_index] - positions[atom1_index]
            vec2[i] = positions[atom2_index] - positions[atom3_index]
        cosine_angle = torch.sum(vec1 * vec2, dim=-1) / (torch.norm(vec1,dim=1) * torch.norm(vec2,dim=1))
        angle_rad = torch.acos(cosine_angle)
        # Use angle_coeffs and angles for your specific system
        angle_energy = (0.5 * self.k_angle * (angle_rad - self.theta0)**2).sum()
        print(((angle_rad - self.theta0)**2).sum())
        return angle_energy

    def dihedral(self, positions,  dihedrals):
        dihedral_energy = 0
        vec1 = torch.zeros((len(dihedrals),3),dtype=float).to(device)
        vec2 = torch.zeros((len(dihedrals),3),dtype=float).to(device)
        vec3 = torch.zeros((len(dihedrals),3),dtype=float).to(device)
        for i,dihedral_quadruplet in enumerate(dihedrals):
            (atom1_type, atom1_index), (atom2_type, atom2_index), (atom3_type, atom3_index), (atom4_type, atom4_index) = dihedral_quadruplet
            vec1[i] = positions[atom2_index] - positions[atom1_index]
            vec2[i] = positions[atom3_index] - positions[atom2_index]
            vec3[i] = positions[atom4_index] - positions[atom3_index]
        normal1 = torch.cross(vec1, vec2)
        normal2 = torch.cross(vec2, vec3)
        cosine_dihedral = torch.sum(normal1 * normal2, dim=-1) / (torch.norm(vec1,dim=1) * torch.norm(vec2,dim=1))
        # dihedral_rad = torch.acos(cosine_dihedral)
        sine_dihedral = torch.sum(torch.cross(normal1, normal2)*vec2,dim=-1) / (torch.norm(vec2,dim=1) * torch.norm(normal1,dim=1) * torch.norm(normal2,dim=1))
        dihedral_rad = torch.atan2(sine_dihedral, cosine_dihedral)
        # Use dihedral_coeffs and dihedrals for your specific system
        # print(dihedral_rad)
        # Use dihedral_coeffs and dihedrals for your specific system
        # dihedral_energy += 0.5 * k_dihedral[0] * (1 + torch.cos( dihedral_rad - delta_phase))+0.5 * k_dihedral[1] * (1 + torch.cos( dihedral_rad *2- delta_phase))+0.5 * k_dihedral[2] * (1 + torch.cos( 3*dihedral_rad - delta_phase))
        dihedral_energy = (0.5 * self.k_dihedral* (1 + torch.cos( self.n*dihedral_rad - self.default_phase))).sum()
        return dihedral_energy

    def Lennard_Jones_potential(self, distances_map, atoms_list):
        lj_energy = 0

        r6 = (self.sigma / distances_map) ** 6
        r12 = r6 * r6
        lj_energy = (4 * self.epsilon * (r12 - r6)).sum()
        return lj_energy

    def coulomb(self, distances_map, atoms_list):
        coulomb_energy = 0
        coulomb_energy = (ke2*(self.charges_pair  / distances_map)).sum()
        # print(coulomb_energy)
        return coulomb_energy

    def forward(self, positions):
        distances_map = self.get_distance_map(positions)
        bond_energy = self.bond(distances_map,  self.bonds)
        angle_energy = self.angle(positions,  self.angles)
        dihedral_energy = self.dihedral(positions, self.dihedrals)
        lj_energy = self.Lennard_Jones_potential(distances_map, self.atoms)
        coulomb_energy = self.coulomb(distances_map, self.atoms)
        
        total_energy = bond_energy + angle_energy + dihedral_energy + lj_energy
        # total_energy = bond_energy + angle_energy + dihedral_energy  
        # total_energy = bond_energy + angle_energy +dihedral_energy + coulomb_energy
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
    energy.backward(retain_graph=True)
    forces = -F_positions.grad
    F_positions.grad.data.zero_()
    return forces

def integrate_Beeman(positions, velocities, masses, forces,cur_forces, dt,energy_model):
    new_positions = positions + dt * velocities + dt * dt * 2/3 * cur_forces/masses-1/6* dt *dt *forces/masses
    new_forces = calculate_forces(new_positions,energy_model)
    new_velocities = velocities + dt * 1/6 * (2 * new_forces/masses + 5 * cur_forces/masses-forces/masses)
    return new_positions, new_velocities, cur_forces, new_forces