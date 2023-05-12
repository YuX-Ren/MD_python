'''
load coeficients from file
'''
import numpy as np
import xml.etree.ElementTree as ET
kcal = 6.9501*1e-5
atom_type_mapping = {"CA": 0,"CB":1,"C": 2, "N": 3,"N2":3, "O":4, "OH":5,"H":6,"H2":6,"H3":6,"H4":6,"H5":6,"HB3":6,"HB2":6,"HB1":6,"HA":6}  # Add other atom types if necessary
n_atom_types = len(atom_type_mapping)
bond_coeffs = np.zeros((n_atom_types, n_atom_types), dtype=float)
bond_lengths = np.zeros((n_atom_types, n_atom_types), dtype=float)
angle_coeffs = np.zeros((n_atom_types, n_atom_types, n_atom_types), dtype=float)
thetas = np.zeros((n_atom_types, n_atom_types, n_atom_types), dtype=float)
dihedral_coeffs = np.zeros((n_atom_types, n_atom_types), dtype=float)
phis = np.zeros((n_atom_types, n_atom_types), dtype=float)
multiplicity = np.zeros((n_atom_types, n_atom_types),  dtype=float)

def parse_forcefield_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for harmonic_bond in root.iter('HarmonicBondForce'):
        for bond in harmonic_bond.iter('Bond'):
            type1 = bond.get('type1')
            type2 = bond.get('type2')
            if type1 not in atom_type_mapping.keys() or type2 not in atom_type_mapping.keys():
                continue
            k = float(bond.get('k'))*kcal*0.01
            length = float(bond.get('length'))*10
            bond_coeffs[atom_type_mapping[type1], atom_type_mapping[type2]] = k
            bond_lengths[atom_type_mapping[type1], atom_type_mapping[type2]] = length


    for harmonic_angle in root.iter('HarmonicAngleForce'):
        for angle in harmonic_angle.iter('Angle'):
            type1 = angle.get('type1')
            type2 = angle.get('type2')
            type3 = angle.get('type3')
            if type1 not in atom_type_mapping.keys() or type2 not in atom_type_mapping.keys() or type3 not in atom_type_mapping.keys():
                continue
            k = float(angle.get('k'))*kcal
            angle_rad = float(angle.get('angle'))
            angle_coeffs[atom_type_mapping[type1],atom_type_mapping[type2], atom_type_mapping[type3]] = k
            thetas[atom_type_mapping[type1],atom_type_mapping[type2], atom_type_mapping[type3]] = angle_rad


    for periodic_torsion in root.iter('PeriodicTorsionForce'):
        for proper in periodic_torsion.iter('Proper'):
            type1 = proper.get('type1')
            type2 = proper.get('type2')
            type3 = proper.get('type3')
            type4 = proper.get('type4')
            if type2 not in atom_type_mapping.keys() or type3 not in atom_type_mapping.keys():
                continue
            k1 = float(proper.get('k1'))*kcal
            periodicity1 = int(proper.get('periodicity1'))
            phase1 = float(proper.get('phase1'))
            
            dihedral_coeffs[atom_type_mapping[type2], atom_type_mapping[type3]] = k1
            phis[atom_type_mapping[type2], atom_type_mapping[type3]] = phase1
            multiplicity[atom_type_mapping[type2], atom_type_mapping[type3]] = periodicity1
    return bond_coeffs,bond_lengths, angle_coeffs, thetas,dihedral_coeffs,phis,multiplicity

bond_coeffs,bond_lengths, angle_coeffs, thetas,dihedral_coeffs,phis,multiplicity = parse_forcefield_xml("FF.xml")

