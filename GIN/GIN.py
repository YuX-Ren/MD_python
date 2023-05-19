## Standard libraries
import os
import json
import math
import numpy as np


## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch_geometric.nn import GINConv, global_add_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GINLayer, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        self_loop_index = torch.stack([torch.arange(0, num_nodes, dtype=torch.long), torch.arange(0, num_nodes, dtype=torch.long)], dim=1).to(x.device)
        edge_index_with_self_loops = torch.cat([edge_index, self_loop_index], dim=0)

        # aggregate neighbor features
        agg_neighbor_feats = torch.zeros([num_nodes, x.size(1)]).to(x.device)
        row, col = edge_index_with_self_loops.t()
        agg_neighbor_feats[row] += x[col]

        out = self.mlp(agg_neighbor_feats)
        return out


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GIN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gin_layers = nn.ModuleList()
        self.gin_layers.append(GINLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gin_layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        out = self.fc(x)
        return out



atom_type_mapping = {"CA": 0,"CB":1,"C": 2, "N": 3, "O":4, "OXT":5,"H2":6,"H3":6,"HB3":6,"HB2":6,"HB1":6,"HA":6,"H":7}  # Add other atom types if necessary
lj_atom_type_mapping = {"CA": 0,"CB":0,"C": 0, "N": 1, "O":2, "OXT":2,"H":3,"H2":3,"H3":3,"HB3":3,"HB2":3,"HB1":3,"HA":3}


n_atom_types = len(atom_type_mapping)

class calculate_forces(nn.Module):
    def __init__(self, universe, layers, dim):
        super(calculate_forces, self).__init__()
        self.atoms,  self.bonds, self.angles, self.dihedrals, self.charges = self.get_topology_data(universe)
        self.embedding = nn.Embedding(n_atom_types, dim)
        self.gin = GIN(32+3, 32, layers)
    def get_topology_data(self, universe):
        atoms_list = torch.tensor([lj_atom_type_mapping[atom.name] for atom in universe.atoms]).to(device)
        bonds = torch.tensor([[bond[0].index, bond[1].index] for bond in universe.bonds]).to(device)

        angles = [[angle[0].index, angle[1].index, angle[2].index] for angle in universe.angles]

        dihedrals = [[dihedral[0].index, dihedral[1].index, dihedral[2].index, dihedral[3].index] for dihedral in universe.dihedrals]

        charges = torch.tensor([atom.charge for atom in universe.atoms]).unsqueeze(dim=1).to(device)
        return atoms_list,  bonds, angles, dihedrals, charges

    def forward(self, positions):
        x = torch.cat([self.embedding(self.atoms),positions], dim=1)
        x = self.gin(x, self.bonds)
        return x

def integrate_Beeman(positions, velocities, masses, forces,cur_forces, dt,forces_func):
    new_positions = positions + dt * velocities + dt * dt * 2/3 * cur_forces/masses-1/6* dt *dt *forces/masses
    new_forces = forces_func(new_positions)
    new_velocities = velocities + dt * 1/6 * (2 * new_forces/masses + 5 * cur_forces/masses-forces/masses)
    return new_positions, new_velocities, cur_forces, new_forces