import os
import pathlib
from os import remove
import logging
from subprocess import Popen
from subprocess import PIPE

from rdkit import Chem
from rdkit.Chem import Descriptors

# mordred
from mordred.error import Missing
from mordred import Calculator, descriptors
import pandas as pd

from GraphXSection.obabel_wrapper import get_feat

logger = logging.getLogger(__name__)
"""Module for calculating molecule-level features to add to the global feature vector."""


def rdkit_descriptors(mol):
    """
    Function to get (all) chemical descriptors from RDKit

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    ret_dict = {}

    # Iterate over all functions that retrieve chemical descriptors
    for name, func in Descriptors.descList:
        ret_dict[name + "_rdkit"] = func(mol)

    return ret_dict


def cdk_descriptors(
    mol, temp_f_smiles_name="tempsmiles.smi", temp_f_cdk_name="tempcdk.txt"
):
    """
    Function to get chemical descriptors from CDK

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    ret_dict = {}

    smiles = Chem.MolToSmiles(mol, 1)

    # Write temporary input file for analysis
    temp_f_smiles = open(temp_f_smiles_name, "w")
    temp_f_smiles.write("%s temp" % smiles)
    temp_f_smiles.close()

    # Get multiple groups of chemical descriptors
    ret_dict.update(
        call_cdk(
            infile=temp_f_smiles_name,
            outfile=temp_f_cdk_name,
            descriptors="topological",
        )
    )
    ret_dict.update(
        call_cdk(
            infile=temp_f_smiles_name, outfile=temp_f_cdk_name, descriptors="geometric"
        )
    )
    ret_dict.update(
        call_cdk(
            infile=temp_f_smiles_name,
            outfile=temp_f_cdk_name,
            descriptors="constitutional",
        )
    )
    ret_dict.update(
        call_cdk(
            infile=temp_f_smiles_name, outfile=temp_f_cdk_name, descriptors="electronic"
        )
    )
    ret_dict.update(
        call_cdk(
            infile=temp_f_smiles_name, outfile=temp_f_cdk_name, descriptors="hybrid"
        )
    )

    # Remove temp files
    remove(temp_f_smiles_name)
    remove(temp_f_cdk_name)

    return ret_dict


def call_rcdk(
    infile_name="temp/tempKnownsl2.csv", outfile_name="temp/tempUnknownsl2.csv"
):
    """
    Get the dataframe associated with this analysis

    Parameters
    ----------
    infile_known : str
        location of a file with known retention time, for Layer 2
    infile_unknown : str
        location of a file with umknown retention time, for Layer 2
    fold_list : str
        the folds to be used in Layer 2

    Returns
    -------
    pd.DataFrame
        test predictions
        pd.DataFrame
                train predictions
    """
    # Write enumerate(ID)+smiles to file -> read file -> execute CDK -> read file here -> todict() -> return
    script_location = os.path.join(pathlib.Path(__file__).parent.resolve(), "CDK_run.R")
    cmd = "Rscript %s %s %s" % (script_location, infile_name, outfile_name)

    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()

    print(out)
    print(err)

    preds = pd.read_csv(outfile_name)
    # Add cdk suffix to the column names

    preds.index = preds["ident"]
    preds.columns = [col + "_cdk" for col in preds.columns]

    return preds[preds.columns.difference(["ident", "SMILES"])].T.to_dict()


def call_cdk(infile="", outfile="", descriptors=""):
    """
    Function to make the call to CDK for chemical descriptor calculation

    Parameters
    ----------
    infile : str
        file to parse the chemical descriptors from
    outfile : str
        temporary out file
    descriptors : str
        type of descriptors to calculate

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    cmd = "java -jar CDKDescUI-1.4.6.jar -b %s -a -t %s -o %s" % (
        infile,
        descriptors,
        outfile,
    )
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)

    # Wait for results ...
    out = p.communicate()

    logging.info(out)

    return parse_cdk_file(outfile)


def parse_cdk_file(file):
    """
    Function to parse CDK output

    Parameters
    ----------
    file : str
        file to parse the chemical descriptors from

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    cdk_file = open(file).readlines()
    cols = cdk_file[0].strip().split()[1:]
    feats = cdk_file[1].strip().split()[1:]
    return dict(zip(cols, feats))


def mordred_descriptors_3D(mol):
    df = pd.DataFrame(mol, index=list(range(len(mol))), columns=["Processed SMILES"])
    logger.debug(df)
    df["index"] = list(range(len(mol)))
    df["Processed SMILES"] = [str(Chem.MolToSmiles(m)) for m in df["Processed SMILES"]]
    logger.debug(df)
    df_feat = get_feat(df, n_jobs=20)
    return df_feat.to_dict()


def mordred_descriptors(mol):
    """
    Function to get chemical descriptors from CDK

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    calc = Calculator(descriptors, ignore_3D=True)
    if type(mol) == list:
        df = calc.pandas(mol).T
        return df.to_dict()
    else:
        df = calc.pandas([mol]).T
        return df.to_dict()[0]


# def rcdk_descriptors(mols):
#     # Create a temporary file to store the SMILES based on __main__ location
#     temp_file_name = os.path.join(
#         pathlib.Path(__file__).parent.resolve(), "temp_smiles.csv"
#     )

#     result_file_name = os.path.join(
#         pathlib.Path(__file__).parent.resolve(), "cdk_features.csv"
#     )
#     temp_file = open(temp_file_name, "w")
#     temp_file.write("ident,SMILES\n")
#     for i, mol in enumerate(mols):
#         temp_file.write("%s,%s\n" % (i, Chem.MolToSmiles(mol, 1)))
#     temp_file.close()

#     # parallelize from here on out
#     return call_rcdk(infile_name=temp_file_name, outfile_name=result_file_name)


def getf(mol, progs=["rdkit"]):
    """
    Main Function to get chemical descriptors from RDKit and/or CDK

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit
    progs : list
        list of programs to run (rdkit or cdk)

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    ret_dict = {}
    if "rdkit" in progs:
        if type(mol) == list:
            ret_dict["rdkit"] = {}
            for i, m in enumerate(mol):
                ret_dict["rdkit"][i] = rdkit_descriptors(m)
        else:
            ret_dict["rdkit"] = rdkit_descriptors(mol)
    # if "cdk" in progs:
    #     ret_dict["cdk"] = rcdk_descriptors(mol)
    #     logger.debug("CDK descriptors: %s", ret_dict["cdk"])
    if "mordred" in progs:
        ret_dict["mordred"] = mordred_descriptors(mol)
    if "mordred3D" in progs:
        ret_dict["mordred3D"] = mordred_descriptors_3D(mol)
    return ret_dict


def get_feats(data, config, ions=["(+H)"], RDKit=False, mordred=False, mordred_3d=True):
    """
    Function to get features (chemical descriptors and chemical class)
    from a csv and put them in a pandas dataframe.

    Parameters
    ----------
    infile : str
        csv file that contains the CCS data to fit the model
    ions : list
        ions to include for the feature extraction
    include_class_vis : list
        chemical classes to include for feature extraction

    Returns
    -------
    object :: pd.DataFrame
        dataframe containing the chemical descriptors
    object :: pd.DataFrame
        dataframe containing the chemical classes
    """
    logger.info("Getting features from data")
    logger.debug("Data shape: %s", data.shape)
    logger.debug("Data: %s", data.head())

    feat_dict = {}

    idents = []
    mols = []

    for row_index, row in data.iterrows():
        smiles = row[config["smiles_col"]]

        identifier = str(row_index) + "|" + row["Chemical Name"]
        mol_obj = Chem.MolFromSmiles(smiles)
        if mol_obj is None:
            logger.warning("Invalid SMILES: %s", smiles)
            continue
        feat_dict[identifier] = {}
        mols.append(mol_obj)
        idents.append(identifier)

        for ion in ions:
            feat_dict[identifier][ion] = 1

    mol_desc = {}
    if RDKit:
        logger.info("Calculating RDKit descriptors")
        mol_desc_rdkit = getf(mols, progs=["rdkit"])["rdkit"]
        logger.info("RDKit descriptors: %s", mol_desc_rdkit)
        for k in mol_desc_rdkit.keys():
            try:
                mol_desc[k].update(mol_desc_rdkit[k])
            except:
                mol_desc[k] = {}
                mol_desc[k].update(mol_desc_rdkit[k])

    if mordred:
        logger.info("Calculating Mordred descriptors")
        mol_desc_mordred = getf(mols, progs=["mordred"])["mordred"]
        for k in mol_desc_mordred.keys():
            try:
                mol_desc[k].update(mol_desc_mordred[k])
            except:
                mol_desc[k] = {}
                mol_desc[k].update(mol_desc_mordred[k])

    if mordred_3d:
        logger.info("Calculating Mordred 3D descriptors")
        mol_desc_mordred_3d = getf(mols, progs=["mordred3D"])["mordred3D"]
        for k in mol_desc_mordred_3d.keys():
            try:
                mol_desc[k].update(mol_desc_mordred_3d[k])
            except:
                mol_desc[k] = {}
                mol_desc[k].update(mol_desc_mordred_3d[k])

    logger.debug(mol_desc.keys())
    for i, ident in enumerate(idents):
        logger.debug("Processing molecule {}/{}: {}".format(i + 1, len(idents), ident))

        mol_desc_key = mol_desc.get(i, mol_desc.get(str(i), {}))
        if not mol_desc_key:
            logger.warning("No descriptors found for index: %s", i)
            continue

        for k, v in mol_desc_key.items():
            try:
                v = float(v)
                if isinstance(v, (float, int)):
                    feat_dict[ident][k] = v
            except ValueError:
                logger.debug("Non-numeric value encountered for key %s: %s", k, v)
                continue

    logger.debug("Feature dictionary keys: %s", feat_dict.keys())
    data_pd = pd.DataFrame(feat_dict).T
    data_pd = data_pd.astype(float)
    # Replace the mordred.error.Missing objects with NaN
    data_pd = data_pd.map(lambda x: float("nan") if isinstance(x, Missing) else x)
    logger.debug(data_pd.index)
    return data_pd


if __name__ == "__main__":
    # Set log level to debug
    logging.basicConfig(level=logging.DEBUG)
    # test_smiles = ["CCCCC", "CCC", "C"]
    # test_mols = [Chem.MolFromSmiles(ts) for ts in test_smiles]
    # res = getf(test_mols, progs=["mordred3D"])
    # print(res)

    # Test get_feats
    test_data = pd.DataFrame(
        {
            "SMILES": ["CCCCC", "CCC", "C"],
            "Chemical Name": ["pentane", "propane", "methane"],
        }
    )
    test_data["Adduct"] = ["(H+)", "(H+)", "(H+)"]
    # Run get_feats
    test_feats = get_feats(
        test_data,
        config={
            "smiles_col": "SMILES",
        },
    )
