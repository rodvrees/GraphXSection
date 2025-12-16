"""Graph encoding utilities for GraphXSection_torch - PyTorch implementation."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from graphamole.data import Molecule
from graphamole.featurizers import ALL_FEATURIZERS

# Import feature configuration system
try:
    from GraphXSection_torch.feature_config import FeatureConfigManager
except ImportError:
    FeatureConfigManager = None

logger = logging.getLogger(__name__)


def get_encoder(config: Dict[str, Any] = None, feature_config_path: Optional[Path] = None):
    """
    Get molecular graph encoder using graphamole.

    This function returns a callable that can convert SMILES strings
    to molecular graph representations.

    Parameters
    ----------
    config : Dict[str, Any], optional
        Configuration dictionary
    feature_config_path : Path, optional
        Path to feature configuration JSON file

    Returns
    -------
    callable
        Function that converts SMILES to graph representations
    """
    logger.info("Creating molecular graph encoder using graphamole")
    
    # Setup feature configuration
    featurizers = ALL_FEATURIZERS  # Default
    
    if feature_config_path is not None and FeatureConfigManager is not None:
        try:
            logger.info(f"Loading custom feature configuration from {feature_config_path}")
            feature_manager = FeatureConfigManager(feature_config_path)
            featurizers = feature_manager.get_featurizer_collection()
            
            # Log vocabulary summary
            vocab_summary = feature_manager.get_vocab_summary()
            logger.info("Custom feature vocabularies loaded:")
            for feat_name, vocab_size in vocab_summary.items():
                logger.info(f"  {feat_name}: {vocab_size} tokens")
                
        except Exception as e:
            logger.warning(f"Failed to load custom feature config: {e}")
            logger.warning("Falling back to default feature configuration")
            featurizers = ALL_FEATURIZERS
    elif feature_config_path is not None:
        logger.warning("Feature configuration requested but FeatureConfigManager not available")
        logger.warning("Falling back to default feature configuration")

    def encode_smiles(smiles_list):
        """
        Encode SMILES strings to molecular graphs.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        List
            List of molecular graph data objects
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        graphs = []
        for smiles in smiles_list:
            try:
                mol = Molecule(smiles)
                graph = mol.get_data(featurizers)
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to encode SMILES {smiles}: {e}")
                # Create a dummy graph for failed encodings
                import torch
                from torch_geometric.data import Data

                dummy_graph = Data(
                    x=torch.zeros(1, 10, dtype=torch.long),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, 3, dtype=torch.long),
                )
                graphs.append(dummy_graph)

        return graphs

    return encode_smiles


class MolecularGraphEncoder:
    """
    Molecular graph encoder class for compatibility with original API.
    """

    def __init__(self, config: Dict[str, Any] = None, feature_config_path: Optional[Path] = None):
        """
        Initialize the molecular graph encoder.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary
        feature_config_path : Path, optional
            Path to feature configuration JSON file
        """
        self.config = config if config is not None else {}
        self.feature_config_path = feature_config_path
        self.encoder_fn = get_encoder(config, feature_config_path)

    def __call__(self, smiles_list):
        """
        Encode SMILES strings to molecular graphs.

        Parameters
        ----------
        smiles_list : List[str] or str
            SMILES string(s) to encode

        Returns
        -------
        List
            List of molecular graph data objects
        """
        return self.encoder_fn(smiles_list)

    def encode_molecules(self, molecules: List[str]):
        """
        Encode a list of molecules.

        Parameters
        ----------
        molecules : List[str]
            List of SMILES strings

        Returns
        -------
        List
            List of molecular graph data objects
        """
        return self(molecules)
