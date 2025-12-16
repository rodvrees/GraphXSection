import subprocess
import os
import re
import tempfile
import argparse
import logging
import sys

from multiprocessing import Pool

import pandas as pd

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem.rdmolfiles import MolFromPDBFile
from multiprocessing import freeze_support
import pandas as pd
from rdkit import Chem

# from obabel_wrapper_batch import parallelize_dataframe, run_df
from mordred import Calculator, descriptors
from argparse import Namespace
from mordred import Calculator, descriptors
from rdkit.Chem.rdmolfiles import MolToMolFile

logger = logging.getLogger(__name__)


def lreplace(pattern, sub, string):
    return re.sub("^%s" % pattern, sub, string)


def generate_3d_mol(inchi_file, outfile_name):
    smiles = open(inchi_file).readlines()[0].strip()
    mol = Chem.MolFromSmiles(smiles)
    MolToMolFile(mol, outfile_name)
    # args = ["obabel","/home/compomics/robbin/CCS/temp/"+inchi_file,"-i","smi", "-o", "mol", "-O","/home/compomics/robbin/CCS/temp/"+outfile_name, "--gen3D"]
    # print(" ".join(args))
    # p = subprocess.Popen(args,shell=True)
    # p.wait()


def generate_3d_mol_obgen(inchi_file, outfile_name):
    args = ["obgen", inchi_file, " > ", outfile_name]
    p = subprocess.Popen(args)
    p.wait()


def generate_conformers_rdkit_inchi(inchi_str, outfile_name, num_conf=2):
    m = Chem.MolFromSmiles(inchi_str)

    try:
        m = Chem.AddHs(m)
    except BaseException:
        pass

    try:
        ids = AllChem.EmbedMultipleConfs(m, numConfs=num_conf, params=AllChem.ETKDG())
    except:
        return False

    writer = Chem.SDWriter(outfile_name)

    for i, conf in enumerate(m.GetConformers()):
        tm = Chem.Mol(m, False, conf.GetId())
        prop = AllChem.MMFFGetMoleculeProperties(tm, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(tm, prop)
        writer.write(tm)

    writer.close()

    return True


def generate_conformers_rdkit(mol_file, outfile_name, num_conf=2):
    m = Chem.MolFromMolFile(mol_file)
    print("Read M")
    try:
        m = Chem.AddHs(m)
    except BaseException:
        pass
    print("Added H")
    ids = AllChem.EmbedMultipleConfs(m, numConfs=num_conf, params=AllChem.ETKDG())
    print("Got multiple confs")
    writer = Chem.SDWriter(outfile_name)

    for i, conf in enumerate(m.GetConformers()):
        tm = Chem.Mol(m, False, conf.GetId())
        prop = AllChem.MMFFGetMoleculeProperties(tm, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(tm, prop)
        writer.write(tm)

    writer.close()


def generate_conformers(mol_file, outfile_name, num_conf=5, score="energy"):
    args = [
        "obabel",
        mol_file,
        "-o",
        "sdf",
        "-O",
        outfile_name,
        "--conformer",
        "--nconf",
        num_conf,
        "--score",
        score,
        "--writefoncormers",
    ]
    args = list(map(str, (args)))
    # print(" ".join(args))
    p = subprocess.Popen(args)
    p.wait()


def split_out_sdf(sdf_file, outfile_names="structure_rmsd_*.mol"):
    args = ["obabel", sdf_file, "-o", "mol", "-O", outfile_names]
    # print(" ".join(args))
    p = subprocess.Popen(args)
    p.wait()


def energy_min_step(
    mol_file, outfile_name, force_field="gaff", min_change_val=1e-04, num_steps=50
):
    args = [
        "obminimize",
        "-sd",
        "-c",
        min_change_val,
        "-o",
        "pdb",
        "-n",
        num_steps,
        "-ff",
        force_field,
        mol_file,
    ]  #        ">",
    # outfile_name
    args = list(map(str, (args)))
    f = open(outfile_name, "w")

    print(" ".join(args))

    p = subprocess.call(args, stdout=f)
    # p = subprocess.Popen(args,shell=True,f)
    # p.wait()
    # p.communicate()
    f.close()


def mol_to_pdb(mol_file, outfile_name):
    args = ["obabel", mol_file, "-o", "pdb", "-O", outfile_name]
    p = subprocess.Popen(args)
    p.wait()


def driftscope(pdb_file, outfile, tolerance=0.0001, gas_val=1, driftscope_loc=""):
    atom_types_loc = os.path.join(driftscope_loc, "docs/atom_types/types.txt")
    args = [
        os.path.join(driftscope_loc, "lib/project.exe"),
        "-pdb",
        pdb_file,
        "-types",
        atom_types_loc,
        "-tol",
        tolerance,
        "-gas",
        gas_val,
        "-out",
        outfile,
    ]
    args = list(map(str, (args)))
    print(" ".join(args))
    p = subprocess.Popen(args)
    p.wait()


def replace_hetatm_with_atm(infile):
    infile_content = open(infile).readlines()
    outfile_content = []
    for line in infile_content:
        outfile_content.append(lreplace("HETATM", "ATOM  ", line))

    outfile = open(infile, "w")
    for line in outfile_content:
        outfile.write(line)

    outfile.close()


def parse_driftscope(infile):
    return max(
        [
            float(line.lstrip("STRUCT_CCS	").rstrip())
            for line in open(infile)
            if line.startswith("STRUCT_CCS")
        ]
    )


def run_pipeline_single_structure(
    identifier,
    inchi_file,
    num_conf=2,
    num_steps=5000,
    run_driftscope=True,
    run_obabel_conformers=True,
    rem_temp_files=True,
    feature_prefix="conformer_",
    driftscope_loc="C:/Users/davy/Desktop/DriftScope/",
    temp_dir="temp/",
):
    ret_dict = {}
    threedmol_filename = temp_dir + "/" + identifier + ".mol"
    conformer_filename = temp_dir + "/" + identifier + "_conformer.sdf"
    splitoutsdf_filename = temp_dir + "/" + identifier + "_conformer_energy_*.mol"
    if run_obabel_conformers:
        logger.debug(
            f"Generating conformers for {inchi_file} with {num_conf} conformers"
        )
        generate_3d_mol(inchi_file, threedmol_filename)
        logger.debug(
            f"Generating conformers for {inchi_file} with {num_conf} conformers and writing to {conformer_filename}"
        )
        generate_conformers(threedmol_filename, conformer_filename, num_conf=num_conf)
        logger.debug(
            f"Generating conformers for {inchi_file} with {num_conf} conformers and writing to {conformer_filename}"
        )
    else:
        logger.debug(
            f"Using RDKit to Generate conformers for {inchi_file} with {num_conf} conformers and writing to {conformer_filename}"
        )
        inchi = open(inchi_file).readlines()[0].rstrip()
        ret_code = generate_conformers_rdkit_inchi(
            inchi, conformer_filename, num_conf=num_conf
        )
        if not ret_code:
            return ret_dict

    split_out_sdf(conformer_filename, outfile_names=splitoutsdf_filename)
    files_for_energy_min = []
    for i in range(num_conf + 1):
        if os.path.exists(splitoutsdf_filename.replace("*", str(i))):
            files_for_energy_min.append(splitoutsdf_filename.replace("*", str(i)))
    for f in files_for_energy_min:
        struct_optimized_filename = f.rstrip(".mol") + "_opt.pdb"
        projection_app_filename = temp_dir + identifier + "_PA.txt"

        energy_min_step(f, struct_optimized_filename, num_steps=num_steps)
        replace_hetatm_with_atm(struct_optimized_filename)
        conformer_num = f.rstrip(".mol").split("_")[-1]

        ret_dict[conformer_num] = 1

        if rem_temp_files:
            # TODO Due to alternative paths files are not always present... Fix
            # this in a better way
            try:
                os.remove(temp_dir + identifier + "_PA.txt")
            except BaseException:
                pass
            try:
                os.remove(f)
            except BaseException:
                pass
            if run_driftscope:
                try:
                    os.remove(f.rstrip(".mol") + "_opt.pdb")
                except BaseException:
                    pass

    if rem_temp_files:
        try:
            os.remove(conformer_filename)
        except BaseException:
            pass
        try:
            os.remove("temp/" + identifier + ".mol")
        except BaseException:
            pass

    for i in range(1, num_conf + 1):
        if feature_prefix + str(i) not in ret_dict.keys():
            ret_dict[feature_prefix + str(i)] = 0.0

    return ret_dict


def run_df(run_tuple):
    df, argu = run_tuple
    df_dict = {}

    for index_name, row in df.iterrows():
        logger.debug(f"Processing {index_name} with SMILES {row['Processed SMILES']}")
        inchi_f_name = str(argu.temp_dir) + str(row["index"]) + ".smi"
        temp_inchi = open(inchi_f_name, "w")
        temp_inchi.write(row["Processed SMILES"])
        temp_inchi.close()

        df_dict[str(row["index"])] = run_pipeline_single_structure(
            str(row["index"]),
            inchi_f_name,
            run_obabel_conformers=argu.run_obabel_conformers,
            num_conf=argu.num_conf,
            rem_temp_files=argu.rem_temp_files,
            run_driftscope=argu.run_driftscope,
            num_steps=argu.num_steps,
            driftscope_loc=argu.driftscope_loc,
            temp_dir=argu.temp_dir,
        )
        if argu.rem_temp_files:
            os.remove(inchi_f_name)
    return pd.DataFrame(df_dict).T


def parallelize_dataframe(
    df, func, temp_dir, cli_arguments, n_cores=4, batch_size=10240
):
    try:
        df_split_first = np.array_split(df, len(df.index) / batch_size)
    except:
        df_split_first = [df]

    for df_index, df_batch in enumerate(df_split_first):
        df_split = np.array_split(df_batch, n_cores)

        run_tuple = list(zip(df_split, [cli_arguments for i in range(n_cores)]))

        pool = Pool(n_cores)
        logger.debug(run_tuple[0])
        df = pd.concat(pool.map(func, run_tuple))
        pool.close()
        pool.join()
        logger.debug(f"Batch {df_index} done")
        df.to_csv("%s/minimized_structs_batch_%s.csv" % (temp_dir, df_index))
    logger.debug("All batches done")
    concat_batches = []
    for f in os.listdir(temp_dir):
        if not f.startswith("minimized_structs_batch"):
            continue
        concat_batches.append(pd.read_csv(os.path.join(temp_dir, f)))
    df = pd.concat(concat_batches, axis=0)

    return df


def parse_argument():
    """
    Read arguments from the command line

    Parameters
    ----------

    Returns
    -------

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--driftscope_loc",
        type=str,
        dest="driftscope_loc",
        default="C:/Users/asus/Desktop/DriftScope/",
        help="Indicate the location of DriftScope for the Projection Approximation approach",
    )

    parser.add_argument(
        "--infile_name",
        type=str,
        dest="infile_name",
        default="CCS_consolidated_sample.csv",
        help="Indicate the location of DriftScope for the Projection Approximation approach",
    )

    parser.add_argument(
        "--outfile_name",
        type=str,
        dest="outfile_name",
        default="ModellingResults.csv",
        help="Indicate the location of DriftScope for the Projection Approximation approach",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        dest="num_steps",
        default=100000,
        help="The maximum number of steps to take for energy minimization",
    )

    parser.add_argument(
        "--num_conf",
        type=int,
        dest="num_conf",
        default=10,
        help="Number of structural conformers to generate",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        dest="n_jobs",
        default=8,
        help="Number of threads to spawn",
    )

    parser.add_argument(
        "--run_driftscope",
        action="store_true",
        help="Flag to indicate to run DriftScope Projection Approximation approach",
    )

    parser.add_argument(
        "--run_obabel_conformers",
        action="store_true",
        help="Flag that indicates if obabel should be used for generating conformers",
    )

    parser.add_argument(
        "--rem_temp_files",
        action="store_true",
        help="Flag that indicates to remove temporary intermediate files",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    results = parser.parse_args()

    return results


def main():
    argu = parse_argument()
    df = pd.read_csv(argu.infile_name, encoding="latin1")

    df_results = parallelize_dataframe(
        df, run_df, cli_arguments=argu, n_cores=argu.n_jobs
    )
    df_results.to_csv(argu.outfile_name)


def get_feat(df, n_jobs=4):
    freeze_support()

    # Make temp directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Temporary directory created: {temp_dir}")

    # df = pd.read_csv("CCS_consolidated_sample.csv", encoding="latin1")
    args = Namespace(
        driftscope_loc="C:/Users/asus/Desktop/DriftScope/",
        infile_name="CCS_consolidated_sample.csv",
        n_jobs=n_jobs,
        num_conf=10,
        num_steps=5000,
        outfile_name="ModellingResults.csv",
        rem_temp_files=True,
        run_driftscope=False,
        run_obabel_conformers=False,
        temp_dir=temp_dir,
    )
    _ = parallelize_dataframe(df, run_df, temp_dir, args, n_cores=n_jobs)

    pdb_files = [
        os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".pdb")
    ]
    mols = [MolFromPDBFile(pdb) for pdb in pdb_files]
    names = [
        f.lstrip(temp_dir).rstrip(".pdb") for f, m in zip(pdb_files, mols) if m != None
    ]
    mols = [m for m in mols if m != None]
    mols = pd.Series(mols, index=names)

    n_all = set(Calculator(descriptors, ignore_3D=False).descriptors)
    n_2D = set(Calculator(descriptors, ignore_3D=True).descriptors)
    features_3D = n_all.difference(n_2D)
    print("calculator")
    print(features_3D)
    # input()
    calc = Calculator(features_3D)
    print("calculator - pandas + ")
    df_ret = calc.pandas(mols, nproc=4)
    df_ret.index = names

    df_ret = df_ret.apply(pd.to_numeric, errors="coerce")
    df_ret.fillna(-1000.0, inplace=True)

    # df_ret = df_ret.loc[:,list(df_ret.dtypes != object)]

    idents = list(set([n.split("_conformer_")[0] for n in df_ret.index]))

    feat_dict = {}
    for i in idents:
        sel = [ident for ident in df_ret.index if ident.startswith(i + "_conformer_")]
        sub_df = df_ret.loc[sel, :]
        sub_df_median = sub_df.median()
        sub_df_min = sub_df.min()
        sub_df_max = sub_df.max()
        sub_df_std = sub_df.std()

        sub_df_median.index = [n + "|median" for n in sub_df_median.index]
        sub_df_min.index = [n + "|min" for n in sub_df_min.index]
        sub_df_max.index = [n + "|max" for n in sub_df_max.index]
        sub_df_std.index = [n + "|std" for n in sub_df_std.index]

        feat_dict[i] = pd.concat(
            [sub_df_median, sub_df_min, sub_df_max, sub_df_std], axis=0
        )

        # if "PPSA1" in sub_df_median.index:
        #    print(feat_dict[i])
        #    input()
    feat_df = pd.DataFrame(feat_dict)

    return feat_df


if __name__ == "__main__":
    freeze_support()

    # Run a single molecule
    run_pipeline_single_structure(
        "test",
        "test.smi",
        num_conf=2,
        num_steps=5000,
        run_driftscope=True,
        run_obabel_conformers=True,
        rem_temp_files=True,
        feature_prefix="conformer_",
        driftscope_loc="C:/Users/davy/Desktop/DriftScope/",
        temp_dir="temp/",
    )
