# GraphXSection
## Training of CCS prediction models for small molecules using Graph Neural Networks

This (very much a WIP) repository contains the code for training and evaluating Graph Neural Network (GNN)-based models for predicting the Collisional Cross Section (CCS) of small molecules.
Furthermore, the creation of saliency maps is also implemented to visually explain the model's prediction.

GraphXSection is an implementation of [Molgraph](https://github.com/akensert/molgraph)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data](#data)

## Installation

To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

## Usage
### Training the model

To train and evaluate a model, prepare a JSON configuration file and run the following command:

```bash
python GraphXSection path_to_config.json
```
## Configuration

The model and training processs are configured via a JSON file. 
Example configuration snippet:
```json
{
    "run_name" : "model_name",
    "data_path" : "path/to/data.csv",
    "output_path" : "path/to/output/dir",
    "valid_split": 0.1,
    "test_split": 0.1,
    
    "atom_encoder" : {
        "symbol" : {
            "allowable_set" : ["C", "N", "O", "H", "F", "P", "I", "Br"],
            "ordinal" : false,
            "oov_size" : 1
        },
        "hybridization" : {
            "allowable_set" : ["SP", "SP2", "SP3"],
            "oov_size" : 0
        },
        "degree" : {
            "allowable_set" : [0,1,2,3,4],
            "oov_size" : 1
        },
        "gasteiger_charge" : true
    },

    "bond_encoder" : {
        "bond_type" : {
            "allowable_set" : ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
        }
    },

    "mcp" : {
        "mcp_path" : "path_to_model_checkpoints/model.keras",
        "mcp_monitor" : "val_loss",
        "mcp_mode" : "min"
    },

    "model" : {
        "type" : "QSAR_linear",
        "optimizer" : "adam",
        "learning_rate" : 0.001,
        "loss" : "mean_absolute_error",
        "metrics" : ["mean_absolute_error"],
        "epochs" : 500,
        "batch_size" : 16,
        "num_GNN_layers" : 3,
        "num_dense_layers" : 3,
        "gnn_units" : 32,
        "use_edge_features" : true,
        "mass_feat": false,
        "dense_units" : 256,
        "normalization" : null,
        "kernel_initializer": null,
        "l1": 0.00001
    },

    "lr_scheduler" : {
        "monitor" : "val_mean_absolute_error",
        "factor" : 0.9999,
        "patience" : 10,
        "min_lr" : 0.00005,
        "min_delta" : 0.01,
        "mode" : "min"
    },

    "use_best_model" : true,
    "save_model" : false,
    "save_dfs" : false,
    "save_predictions" : true,
    "save_plot" : false,
    "saliency_mapping" : true,
    "wandb" : {
        "use" : true,
        "project" : "GraphCCS"
    }
}
```
## Data

The program expects CSV-formatted data, which should be specified in the `data_path` configuration parameter within your JSON configuration file. The CSV file should contain at least the following columns:

- **SMILES:** The SMILES (Simplified Molecular Input Line Entry System) representation of the molecule. This is a string that encodes the molecular structure and is used as input to the molecular encoder.
- **Adduct:** The type of adduct (e.g., `(+H)`) associated with the molecule. This is crucial for accurate CCS prediction as different adducts can affect the CCS values.
- **monoisotopic mass:** The monoisotopic mass of the molecule, for plotting purposes.
- **CCS:** The experimentally determined Collisional Cross Section value that the model aims to predict. This is the target variable for training the model.

### Example Data Format

```csv
SMILES,Adduct,monoisotopic mass,CCS
C1=CC=CC=C1,(+H),78.046950,150.2
C1=CC(=CC=C1C=O)O,(+Na),136.031300,170.5
CC(C)C1=CC=C(C=C1)O,(+K),150.080400,190.3


