"""Module for calculating molecule-level features to add to the global feature vector."""

import sys

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA

# ### Structural and Shape Features

# 1. Molecular Weight
# 2. Molecual Volume (Van der Waals Volume, rdKit’s rdFreeSASA)
# 3. Radius of Gyration
# 4. Principal Moments of Inertia (PMI1, PMI2, PMI3)
# 5. Asphericity
# 6. SASA

# ### Flexibility and Conformational Freedom

# 1. Number of Rotatable Bonds
# 2. Fraction of sp3 hybridized Carbons
# 3. Number of cycles and ring strain energy

# ### Electronic and Charge Distribution

# 1. Dipole Moment
# 2. Electrostatic potential Surface Area
# 3. HOMO-LUMO Gap
# 4. Polarizablity

# Atomic polarizability values from 2018 Table of Static Dipole Polarizabilities (Mol. Phys. 2019)
ATOMIC_POLARIZABILITY = {
    "H": 0.09,
    "He": 0.02,
    "Li": 9.10,
    "Be": 1.34,
    "B": 1.51,
    "C": 0.73,
    "N": 0.52,
    "O": 0.54,
    "F": 0.46,
    "Ne": 0.43,
    "Na": 11.80,
    "Mg": 3.18,
    "Al": 5.23,
    "Si": 2.19,
    "P": 1.34,
    "S": 1.30,
    "Cl": 1.02,
    "Ar": 0.91,
    "K": 19.69,
    "Ca": 6.22,
    "Sc": 5.09,
    "Ti": 4.49,
    "V": 4.48,
    "Cr": 4.33,
    "Mn": 3.48,
    "Fe": 3.07,
    "Co": 3.01,
    "Ni": 3.12,
    "Cu": 3.01,
    "Zn": 2.24,
    "Ga": 4.47,
    "Ge": 2.55,
    "As": 1.97,
    "Se": 1.90,
    "Br": 1.67,
    "Kr": 1.56,
    "Rb": 22.14,
    "Sr": 7.60,
    "Y": 5.57,
    "Zr": 4.63,
    "Nb": 4.45,
    "Mo": 4.08,
    "Tc": 3.80,
    "Ru": 3.64,
    "Rh": 3.49,
    "Pd": 2.87,
    "Ag": 3.29,
    "Cd": 2.54,
    "In": 5.01,
    "Sn": 3.02,
    "Sb": 2.37,
    "Te": 2.22,
    "I": 1.97,
    "Xe": 1.82,
    "Cs": 26.99,
    "Ba": 11.07,
    "Hf": 3.19,
    "Ta": 2.47,
    "W": 2.40,
    "Re": 2.40,
    "Os": 2.15,
    "Ir": 2.05,
    "Pt": 2.05,
    "Au": 2.00,
    "Hg": 1.85,
    "Tl": 7.00,
    "Pb": 4.16,
    "Bi": 4.04,
    "Po": 2.98,
    "At": 2.40,
    "Rn": 2.11,
}


def process_molecule(smiles):
    """Preprocess, embed, and optimize a molecule from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Skipping invalid SMILES: {smiles}")
        return None

    mol = Chem.AddHs(mol)  # Add hydrogens

    # Try ETKDG embedding first, fallback to random coordinates
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            return None  # Skip if both embedding methods fail

    # Optimize structure
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            return None  # Skip if both fail

    # Ensure molecule has a valid conformer
    if mol.GetNumConformers() == 0:
        return None

    return mol


def calculate_features_from_mol(mol):
    """Calculate all molecular features for a single molecule (preprocessed)."""
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    # **1. Molecular Weight**
    molecular_weight = Descriptors.ExactMolWt(mol)

    # **2. Van der Waals Volume (SASA)**
    for atom in mol.GetAtoms():
        atom.SetDoubleProp(
            "SASA_radius", Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
        )
    try:
        radii = rdFreeSASA.classifyAtoms(mol)
        vdw_volume = rdFreeSASA.CalcSASA(mol, radii)
    except:
        vdw_volume = None

    # **3. Radius of Gyration**
    coords = np.array([conf.GetAtomPosition(i) for i in range(num_atoms)])
    center_of_mass = np.mean(coords, axis=0)
    radius_of_gyration = np.sqrt(
        np.mean(np.sum((coords - center_of_mass) ** 2, axis=1))
    )

    # **4. Principal Moments of Inertia (PMI)**
    atomic_weights = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    inertia_tensor = np.zeros((3, 3))
    for i in range(num_atoms):
        x, y, z = coords[i] - center_of_mass
        m = atomic_weights[i]
        inertia_tensor[0, 0] += m * (y**2 + z**2)
        inertia_tensor[1, 1] += m * (x**2 + z**2)
        inertia_tensor[2, 2] += m * (x**2 + y**2)
        inertia_tensor[0, 1] -= m * x * y
        inertia_tensor[0, 2] -= m * x * z
        inertia_tensor[1, 2] -= m * y * z
    inertia_tensor[1, 0], inertia_tensor[2, 0], inertia_tensor[2, 1] = (
        inertia_tensor[0, 1],
        inertia_tensor[0, 2],
        inertia_tensor[1, 2],
    )
    pmi1, pmi2, pmi3 = np.linalg.eigvalsh(inertia_tensor)

    # **5. Asphericity**
    asphericity = ((pmi1 - pmi2) ** 2 + (pmi2 - pmi3) ** 2 + (pmi3 - pmi1) ** 2) / (
        2 * (pmi1 + pmi2 + pmi3) ** 2
    )

    # **6. Dipole Moment (Using Gasteiger Charges)**
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    dipole_vector = np.sum(
        [
            float(atom.GetProp("_GasteigerCharge")) * np.array(conf.GetAtomPosition(i))
            for i, atom in enumerate(mol.GetAtoms())
        ],
        axis=0,
    )
    dipole_magnitude = np.linalg.norm(dipole_vector)

    # **7. Polarizability**
    polarizability = sum(
        ATOMIC_POLARIZABILITY.get(atom.GetSymbol(), 0) for atom in mol.GetAtoms()
    )

    return [
        molecular_weight,
        vdw_volume,
        radius_of_gyration,
        asphericity,
        dipole_magnitude,
        polarizability,
    ]


# TODO make it different if no features in the dataframe
def calculate_features(df, smiles_col):
    """Calculate molecular features for an entire dataset."""
    adducts = df["Adduct"]
    df["MOL"] = df[smiles_col].apply(process_molecule)
    df["Features"] = df["MOL"].apply(calculate_features_from_mol)
    print("First few features before scaling: ", df["Features"][:5])
    df.drop(columns=["MOL"], inplace=True)  # Remove processed molecules to save memory

    features = np.array(df["Features"].tolist())
    return features


if __name__ == "__main__":
    # Test the feature calculation
    # df = pd.DataFrame(
    #     {
    #         "SMILES": [
    #             "CCO",
    #             "C1=CC=CC=C1",
    #             "CC(=O)O",
    #             "CC(C)(C)C",
    #             "C1=CC=CC=C1",
    #         ],
    #         "Adduct": ["H+", "Na+", "K+", "H+", "H+"],
    #     }
    # )
    # features = calculate_features(df, "SMILES", train=True)
    # print(features)

    df = pd.read_csv(
        "/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/data/protonated_data_2522025.csv"
    )

    features = calculate_features(df, "protonated_SMILES")

    # Add the features to the dataframe, each in a separate column
    feature_names = [
        "Molecular Weight",
        "Van der Waals Volume",
        "Radius of Gyration",
        "Asphericity",
        "Dipole Moment",
        "Polarizability",
    ]

    for i, feature in enumerate(feature_names):
        df[feature] = features[:, i]

    # Save the features to a CSV file
    df.to_csv(
        "/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/data/protonated_data_2522025_features.csv",
        index=False,
    )
    print("Features saved to CSV")
