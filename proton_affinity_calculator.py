import os
import multiprocessing
import shutil
import copy
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import logging
from tqdm import tqdm
from multiprocessing import Lock
tqdm.pandas()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lock = Lock()

def generate_3d_structure(smiles):
    """Generate a 3D structure from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add explicit hydrogens
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D coordinates
    AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry with UFF
    return mol

def calculate_total_electrons(mol, charge):
    """Calculate the total number of electrons in the molecule."""
    total_electrons = 0
    for atom in mol.GetAtoms():
        total_electrons += atom.GetAtomicNum() - atom.GetFormalCharge()
    return total_electrons

def identify_protonation_sites(mol):
    """Identify potential protonation sites in the molecule."""
    potential_sites = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8, 16,]: # N, O, S  
            potential_sites.append(atom.GetIdx())
    return potential_sites

def protonate_molecule(mol, atom_idx):
    """Protonate the molecule at the specified atom index."""
    mol_protonated = copy.deepcopy(mol)
    atom = mol_protonated.GetAtomWithIdx(atom_idx)
    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
    
    atom.UpdatePropertyCache()

    mol_protonated = Chem.AddHs(mol_protonated)

    # Generate 3D coordinates for the protonated molecule
    AllChem.EmbedMolecule(mol_protonated, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol_protonated)
    return mol_protonated

def write_xyz_file(mol, filename):
    """Write RDKit molecule to an XYZ file."""
    Chem.MolToXYZFile(mol, filename)

def write_orca_input(xyz_filename, input_filename, charge=0, multiplicity=1, processes=1):
    """Write an ORCA input file."""
    with open(input_filename, 'w') as f:
        f.write(f"! PBE def2-SVP RI Energy\n")
        f.write(f"%geom MaxIter 50 end\n")
        f.write(f"%output\n  Print[ P_Basis ] 2\nend\n")
        f.write(f"%pal nprocs {processes} end\n")
        f.write(f"* xyz {charge} {multiplicity}\n")
        with open(xyz_filename, 'r') as xyz_file:
            lines = xyz_file.readlines()[2:]  # Skip first two lines
            f.writelines(lines)
        f.write("*\n")

def run_orca(input_filename, output_filename):
    """Run ORCA calculation."""
    # Adjust the path to the ORCA executable if necessary
    orca_executable = "orca"
    try:
        with open(output_filename, 'w') as outfile:
            subprocess.run([orca_executable, input_filename], stdout=outfile, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        logger.info(f"ORCA calculation failed for {input_filename}")
        return None

def parse_orca_output(output_filename):
    """Parse ORCA output file to extract energy."""
    energy = None
    with open(output_filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'FINAL SINGLE POINT ENERGY' in line:
                energy_line = line.strip()
                energy = float(energy_line.split()[-1])  # In Hartree
    if energy is None:
        logger.info(f"Could not find energy in {output_filename}")
        return None
    return energy

def process_molecule(smiles, processes=1):
    """Process a molecule and return the most likely protonated molecule's SMILES."""
    try:
        mol = generate_3d_structure(smiles)
        if mol is None:
            return 'COULD NOT GENERATE 3D structure from SMILES', None

        # Create a unique temp directory for each molecule
        import tempfile
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Temp directory for molecule {smiles}: {temp_dir}")

        # Identify potential protonation sites
        potential_sites = identify_protonation_sites(mol)
        if not potential_sites:
            logger.info(f"No potential protonation sites found for molecule: {smiles}")
            return 'NO PROTONATION SITES', None
        
        if len(potential_sites) == 1:
            logger.info(f"Only one potential protonation site found for molecule: {smiles}")
            best_idx = potential_sites[0]
            best_protonated_mol = protonate_molecule(mol, best_idx)
            protonated_smiles = Chem.MolToSmiles(best_protonated_mol)
            # Single site, so only one proton affinity
            proton_affinities = [{'site': best_idx, 'atom': mol.GetAtomWithIdx(best_idx).GetSymbol(), 'PA': 'Only one site'}]
            return protonated_smiles, proton_affinities

        # Write XYZ file for the neutral molecule
        neutral_xyz = os.path.join(temp_dir, "neutral.xyz")
        write_xyz_file(mol, neutral_xyz)

        # Calculate charge and multiplicity for the neutral molecule
        neutral_charge = 0
        total_electrons = calculate_total_electrons(mol, neutral_charge)
        neutral_multiplicity = 1 if total_electrons % 2 == 0 else 2

        # Write ORCA input file for the neutral molecule
        neutral_inp = os.path.join(temp_dir, "neutral.inp")
        write_orca_input(neutral_xyz, neutral_inp, charge=neutral_charge, multiplicity=neutral_multiplicity, processes=processes)

        # Run ORCA calculation for the neutral molecule
        os.chdir(temp_dir)
        neutral_out = "neutral.out"
        try:
            run_orca("neutral.inp", neutral_out)
            neutral_energy = parse_orca_output(neutral_out)
            if neutral_energy is None:
                return 'COULD NOT CALCULATE NEUTRAL ENERGY', None
            logger.info(f"Neutral molecule energy for {smiles}: {neutral_energy:.6f} Hartree")
        except Exception as e:
            logger.info(f"Failed to calculate neutral molecule energy for {smiles}: {e}")
            return None, None
        finally:
            os.chdir("..")

        proton_affinities = []
        for idx in potential_sites:
            atom = mol.GetAtomWithIdx(idx)
            atom_symbol = atom.GetSymbol()
            logger.info(f"Processing protonation at atom index {idx} ({atom_symbol}) for molecule: {smiles}")

            # Generate protonated molecule
            mol_protonated = protonate_molecule(mol, idx)

            # Write XYZ file for the protonated molecule
            protonated_xyz = os.path.join(temp_dir, f"protonated_{idx}.xyz")
            write_xyz_file(mol_protonated, protonated_xyz)

            # Calculate charge and multiplicity for the protonated molecule
            protonated_charge = neutral_charge + 1
            total_electrons = calculate_total_electrons(mol_protonated, protonated_charge)
            protonated_multiplicity = 1 if total_electrons % 2 == 0 else 2

            # Write ORCA input file for the protonated molecule
            protonated_inp = os.path.join(temp_dir, f"protonated_{idx}.inp")
            write_orca_input(protonated_xyz, protonated_inp, charge=protonated_charge, multiplicity=protonated_multiplicity, processes=processes)

            # Run ORCA calculation for the protonated molecule
            os.chdir(temp_dir)
            protonated_out = f"protonated_{idx}.out"
            try:
                run_orca(f"protonated_{idx}.inp", protonated_out)
                protonated_energy = parse_orca_output(protonated_out)
                if protonated_energy is None:
                    return 'COULD NOT CALCULATE PROTONATED ENERGY', None
                logger.info(f"Protonated molecule energy at site {idx} for {smiles}: {protonated_energy:.6f} Hartree")
                # Proton affinity calculation
                PA = -(protonated_energy - neutral_energy) * 627.5095  # Convert Hartree to kcal/mol
                logger.info(f"Proton affinity at site {idx} for {smiles}: {PA:.6f} kcal/mol")
                proton_affinities.append({'site': idx, 'atom': atom_symbol, 'PA': PA})
            except Exception as e:
                logger.error(f"Failed to calculate protonated molecule energy at site {idx} for {smiles}: {e}")
                continue
            finally:
                os.chdir("..")

        # Clean up the temp directory
        shutil.rmtree(temp_dir)

        if not proton_affinities:
            logger.info(f"Failed to calculate proton affinities for any site in molecule: {smiles}")
            return None, None

        # Determine the site with the highest proton affinity
        proton_affinities.sort(reverse=True, key=lambda x: x['PA'])
        highest_PA = proton_affinities[0]['PA']
        best_sites = [entry for entry in proton_affinities if entry['PA'] == highest_PA]
        
        if len(best_sites) > 1:
            logger.info(f"Multiple sites with the highest proton affinity for {smiles}: {best_sites}")
            return 'UNCLEAR', proton_affinities
        
        else:
            best_idx = best_sites[0]['site']
            best_atom_symbol = best_sites[0]['atom']

            logger.info(f"Most likely protonation site for {smiles}: Atom index {best_idx} ({best_atom_symbol}) with proton affinity {highest_PA:.2f} kcal/mol")

            # Generate the most likely protonated molecule
            best_protonated_mol = protonate_molecule(mol, best_idx)
            # Convert to SMILES
            protonated_smiles = Chem.MolToSmiles(best_protonated_mol)
            return protonated_smiles, proton_affinities

    except Exception as e:
        logger.info(f"Error processing molecule {smiles}: {e}")
        return None, None
    finally:
        # Ensure temp_dir is cleaned up
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_molecule_wrapper(row):
    """Wrapper function to unpack row and call process_molecule for multiprocessing."""
    smiles = row['SMILES']
    result = {'SMILES': smiles}
    try:
        protonated_smiles, proton_affinities = process_molecule(smiles)
        result['protonated_SMILES'] = protonated_smiles
        result['proton_affinities'] = proton_affinities
    except Exception as e: 
        logger.info(f"Error processing molecule {smiles}: {e}")
        result['protonated_SMILES'] = None
        result['proton_affinities'] = None
    return result

def write_result_to_csv(result, output_file, write_header=False):
    """Write the result to a CSV file."""
    with lock:
        result_df = pd.DataFrame([result])
        result_df.to_csv(output_file, header=write_header, index=False, mode='a')
    
def main():
    data = pd.read_csv('/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/data/CCS_consolidated_SMILES_processed.csv')
    output_file = '/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/data/CCS_consolidated_SMILES_processed_protonated.csv'

    # Filter out duplicate SMILES to avoid redundant calculations, afterwards we will merge the results back
    unique_smiles_df = data.drop_duplicates(subset='SMILES')

    write_header = not os.path.exists(output_file)

    # Set up multiprocessing pool
    num_processes = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.imap(process_molecule_wrapper, unique_smiles_df.to_dict(orient='records'))

        for result in tqdm(results, total=unique_smiles_df.shape[0], desc="Processing molecules"):
            write_result_to_csv(result, output_file, write_header)
            write_header = False

if __name__ == '__main__':
    main()
