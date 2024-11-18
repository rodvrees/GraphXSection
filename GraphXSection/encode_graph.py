"""Configure molecular encoder for GraphXSection model"""

import logging
from molgraph import chemistry
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Atom feature mapping
ATOM_FEATURES = {
    "symbol": chemistry.features.Symbol,
    "chiral_center": chemistry.features.ChiralCenter,
    "chirality": chemistry.features.CIPCode,
    "logP": chemistry.features.CrippenLogPContribution,
    "refractivity": chemistry.features.CrippenMolarRefractivityContribution,
    "degree": chemistry.features.Degree,
    "formal_charge": chemistry.features.FormalCharge,
    "gasteiger_charge": chemistry.features.GasteigerCharge,
    "hybridization": chemistry.features.Hybridization,
    "is_aromatic": chemistry.features.Aromatic,
    "H_donor": chemistry.features.HydrogenDonor,
    "H_acceptor": chemistry.features.HydrogenAcceptor,
    "hetero": chemistry.features.Hetero,
    "in_ring": chemistry.features.Ring,
    "in_ring_n": chemistry.features.RingSize,
    "labute_asa": chemistry.features.LabuteASAContribution,
    "n_H": chemistry.features.TotalNumHs,
    "n_rad_e": chemistry.features.NumRadicalElectrons,
    "n_val_e": chemistry.features.TotalValence,
    "topo_pol_sasa": chemistry.features.TPSAContribution,
}

# Bond feature mapping
BOND_FEATURES = {
    "bond_type": chemistry.features.BondType,
    "conjugated": chemistry.features.Conjugated,
    "rotatable": chemistry.features.Rotatable,
    "stereo": chemistry.features.Stereo,
}


def _get_atom_features(config: Dict[str, Any]) -> List[chemistry.features.Feature]:
    """
    Extract and configure atom features for the molecular encoder based on the given configuration

    Args:
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        List[chemistry.features.Feature]: List of atom features to be used in the molecular encoder

    Raises:
        ValueError: If an invalid atom feature is provided in the configuration
    """

    model_atom_features = []
    atom_enc_config = config["atom_encoder"]
    for feature, value in atom_enc_config.items():
        if feature not in ATOM_FEATURES:
            raise ValueError(f"Invalid atom feature: {feature}")
        else:
            # If value is a boolean, add the feature if True
            if isinstance(value, bool):
                if value:
                    model_atom_features.append(ATOM_FEATURES[feature]())
            # If value is a dictionary, then pass the arguments to the feature
            else:
                logger.debug(f"Adding atom feature: {feature}")
                model_atom_features.append(ATOM_FEATURES[feature](**value))
    return model_atom_features


def _get_bond_features(config: Dict[str, Any]) -> List[chemistry.features.Feature]:
    """
    Extract and configure bond features for the molecular encoder based on the given configuration

    Args:
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        List[chemistry.features.Feature]: List of bond features to be used in the molecular encoder

    Raises:
        ValueError: If an invalid bond feature is provided in the configuration
    """

    model_bond_features = []
    bond_enc_config = config["bond_encoder"]
    for feature, value in bond_enc_config.items():
        if feature not in BOND_FEATURES:
            raise ValueError(f"Invalid bond feature: {feature}")
        else:
            # If value is a boolean, then add the feature if True
            if isinstance(value, bool):
                if value:
                    model_bond_features.append(BOND_FEATURES[feature]())
            # If value is a dictionary, then pass the arguments to the feature
            else:
                model_bond_features.append(BOND_FEATURES[feature](**value))
    return model_bond_features


def get_encoder(config: Dict[str, Any]) -> chemistry.MolecularGraphEncoder:
    """
    Configure the molecular encoder based on the given configuration

    Args:
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        chemistry.MolecularGraphEncoder: Molecular encoder object
    """

    logger.info("Setting up molecular encoder")
    atom_features = _get_atom_features(config)
    bond_features = _get_bond_features(config)
    atom_encoder = chemistry.Featurizer(atom_features)
    try:
        bond_encoder = chemistry.Featurizer(bond_features)
        if not config["mol_encoder"].get("3D", False):
            logger.debug("Setting up 2D molecular encoder")
            mol_encoder = chemistry.MolecularGraphEncoder(
                atom_encoder,
                bond_encoder,
                positional_encoding_dim=config["mol_encoder"][
                    "positional_encoding_dim"
                ],
            )

        else:
            raise NotImplementedError("3D molecular encoder is not implemented yet")
    except ValueError:
        logger.debug("No bond features provided.")
        if not config["mol_encoder"].get("3D", False):
            logger.debug("Setting up 2D molecular encoder")
            mol_encoder = chemistry.MolecularGraphEncoder(
                atom_encoder,
                bond_encoder=None,
                positional_encoding_dim=config["mol_encoder"][
                    "positional_encoding_dim"
                ],
            )
        else:
            raise NotImplementedError("3D molecular encoder is not implemented yet")

    logger.info("Molecular encoder setup complete")
    return mol_encoder
