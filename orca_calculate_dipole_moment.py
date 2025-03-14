"""
Parallelized ORCA dipole moment calculation using multiprocessing.
"""

import os
import sys
import subprocess
import tempfile
import pandas as pd
import numpy as np
from multiprocessing import Pool, Lock, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem
import threading

# Global lock for safe file writes
lock = threading.Lock()


def write_orca_input_header(orca_input_file):
    """Write ORCA input file header."""
    with open(orca_input_file, "w") as f:
        f.write("! B3LYP DEF2-SVP TightSCF\n")


def write_orca_task(orca_input_file):
    """Write ORCA task section."""
    with open(orca_input_file, "a") as f:
        f.write("%elprop\nDipole 1\nPolar 1\nend\n")


def calculate_multiplicity(smiles):
    """Calculate molecular multiplicity."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1  # Default singlet

    num_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    return num_radicals + 1  # Multiplicity = 2S + 1


def write_orca_coordinate_header(orca_input_file, charge, multiplicity):
    """Write charge and multiplicity information."""
    with open(orca_input_file, "a") as f:
        f.write(f"* xyz {charge} {multiplicity}\n")


def write_coords(orca_input_file, smiles):
    """Generate 3D coordinates and write to ORCA input file."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            return False  # Failed to generate structure

    conf = mol.GetConformer()
    with open(orca_input_file, "a") as f:
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}\n")
        f.write("*\n")
    return True


def run_orca(orca_input_file, orca_output_file):
    """Run ORCA simulation."""
    orca_command = f"orca {orca_input_file} > {orca_output_file}"
    subprocess.run(
        orca_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def parse_orca_output(orca_output_file):
    """Extract dipole moment and polarizability from ORCA output."""
    dipole_moment, polarizability = np.nan, np.nan  # Default to NaN

    with open(orca_output_file, "r") as f:
        for line in f:
            if "Magnitude (Debye)" in line:
                dipole_moment = float(line.split()[3])
            if "Isotropic polarizability" in line:
                polarizability = float(line.split()[3])

    return dipole_moment, polarizability


def process_molecule(args):
    """Parallelized function to process each molecule."""
    i, smi, charge, output_csv, already_processed = args

    if i in already_processed:
        print(f"[{i+1}] Skipping already processed molecule: {smi}")
        return None

    print(f"[{i+1}] Processing molecule: {smi}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        orca_input_file = os.path.join(tmpdirname, f"orca_input_{i}.inp")
        orca_output_file = os.path.join(tmpdirname, f"orca_output_{i}.out")

        # Write ORCA input
        write_orca_input_header(orca_input_file)
        write_orca_task(orca_input_file)
        multiplicity = calculate_multiplicity(smi)
        write_orca_coordinate_header(orca_input_file, charge, multiplicity)
        if not write_coords(orca_input_file, smi):
            print(f"[{i+1}] Failed to generate coordinates for {smi}")
            return None  # Skip molecule

        # Run ORCA
        run_orca(orca_input_file, orca_output_file)

        # Parse results
        dipole_moment, polarizability = parse_orca_output(orca_output_file)

        # Lock file access and write results safely
        with lock:
            df = pd.read_csv(output_csv)  # Read CSV safely
            df.loc[i, "Dipole moment Orca"] = dipole_moment
            df.loc[i, "Polarizability Orca"] = polarizability
            df.to_csv(output_csv, index=False)  # Overwrite safely

        print(
            f"[{i+1}] Done - Dipole: {dipole_moment} Debye, Polarizability: {polarizability} a.u."
        )
    return i


def main(input_csv, output_csv, num_workers=None):
    """Main function to parallelize ORCA calculations."""

    # Read dataset
    df = pd.read_csv(input_csv)

    # Assign charges based on adducts
    df["Charge"] = df["Adduct"].map(lambda x: -1 if x == "(-H)" else 1)

    # Ensure output file exists
    if os.path.exists(output_csv):
        df_out = pd.read_csv(output_csv)
        if "Dipole moment Orca" not in df_out.columns:
            df_out["Dipole moment Orca"] = np.nan
        if "Polarizability Orca" not in df_out.columns:
            df_out["Polarizability Orca"] = np.nan
    else:
        df_out = df.copy()
        df_out["Dipole moment Orca"] = np.nan
        df_out["Polarizability Orca"] = np.nan
        df_out.to_csv(output_csv, index=False)  # Initialize output CSV

    # Identify already processed molecules
    already_processed = set(df_out.dropna(subset=["Dipole moment Orca"]).index)

    not_processed = df[~df.index.isin(already_processed)]

    # Move not_processed rows from df to df_out
    df_out = pd.concat([df_out, not_processed], ignore_index=True)
    df_out.to_csv(output_csv, index=False)

    # Prepare arguments for multiprocessing
    args_list = [
        (i, row["protonated_SMILES"], row["Charge"], output_csv, already_processed)
        for i, row in df.iterrows()
    ]

    # Use all available CPUs unless specified
    num_workers = num_workers or max(1, cpu_count() - 1)

    print(f"Starting ORCA calculations with {num_workers} workers...\n")

    # Parallel processing using multiprocessing
    with Pool(processes=num_workers) as pool:
        pool.map(process_molecule, args_list)

    print("All calculations complete. Results saved to:", output_csv)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py input.csv output.csv [num_workers]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    main(input_csv, output_csv, num_workers)
