"""Feature configuration system for GraphXSection_torch."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from copy import deepcopy
import sys
import os

# Add graphamole to path - handle multiple possible locations
current_dir = os.path.dirname(__file__)
possible_paths = [
    os.path.join(current_dir, "..", "graphamole"),
    os.path.join(current_dir, "..", "graphamole", "graphamole"),
    os.path.abspath(os.path.join(current_dir, "..", "graphamole")),
    os.path.abspath(os.path.join(current_dir, "..", "graphamole", "graphamole")),
]

for path in possible_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from graphamole.featurizers import atom, bond, collection
    from graphamole.featurizers.atom import DEFAULT_VOCABULARY as ATOM_DEFAULT_VOCAB
    from graphamole.featurizers.bond import DEFAULT_VOCABULARY as BOND_DEFAULT_VOCAB

    GRAPHAMOLE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Failed to import graphamole: {e}")
    GRAPHAMOLE_AVAILABLE = False
    # Create dummy classes for when graphamole is not available
    atom = None
    bond = None
    collection = None
    ATOM_DEFAULT_VOCAB = {}
    BOND_DEFAULT_VOCAB = {}

logger = logging.getLogger(__name__)


class FeatureConfigManager:
    """Manages feature configuration and vocabulary customization."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize feature configuration manager.

        Args:
            config_path: Path to feature configuration JSON file
        """
        if not GRAPHAMOLE_AVAILABLE:
            raise ImportError(
                "Graphamole is not available. Please install or check the path."
            )

        self.config_path = config_path
        self.custom_vocab = {}
        self.atom_featurizers = []
        self.bond_featurizers = []

        if config_path is not None:
            self.load_config(config_path)
        else:
            self._setup_default_config()

    def load_config(self, config_path: Path) -> None:
        """
        Load feature configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Expected JSON structure:
        {
            "vocabularies": {
                "AtomType": ["C", "N", "O", "S", ...],
                "BondType": ["SINGLE", "DOUBLE", "TRIPLE", ...],
                ...
            },
            "atom_features": [
                {"type": "AtomType", "oov": true},
                {"type": "Hybridization", "oov": false},
                ...
            ],
            "bond_features": [
                {"type": "BondType", "oov": true},
                {"type": "Stereo", "oov": false},
                ...
            ]
        }
        """
        logger.info(f"Loading feature configuration from {config_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Feature config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Load custom vocabularies
        self.custom_vocab = config.get("vocabularies", {})
        logger.info(f"Loaded {len(self.custom_vocab)} custom vocabularies")

        # Setup atom featurizers
        atom_config = config.get("atom_features", [])
        self.atom_featurizers = self._create_atom_featurizers(atom_config)
        logger.info(f"Created {len(self.atom_featurizers)} atom featurizers")

        # Setup bond featurizers
        bond_config = config.get("bond_features", [])
        self.bond_featurizers = self._create_bond_featurizers(bond_config)
        logger.info(f"Created {len(self.bond_featurizers)} bond featurizers")

    def _setup_default_config(self) -> None:
        """Setup default featurizers using DEFAULT_VOCABULARY."""
        logger.info("Setting up default feature configuration")

        # Default atom featurizers (matching ALL_FEATURIZERS)
        self.atom_featurizers = [
            atom.AtomType(),
            atom.Hybridization(),
            atom.CIPCode(),
            atom.FormalCharge(),
            atom.TotalNumHs(),
            atom.TotalValence(),
            atom.NumRadicalElectrons(),
            atom.Degree(),
            atom.RingSize(),
            atom.ChiralCenter(),
            atom.Aromatic(),
            atom.Hetero(),
            atom.HydrogenDonor(),
            atom.HydrogenAcceptor(),
            atom.Ring(),
        ]

        # Default bond featurizers
        self.bond_featurizers = [
            bond.BondType(),
            bond.Stereo(),
            bond.Conjugated(),
            bond.Rotatable(),
        ]

    def _create_atom_featurizers(self, atom_config: List[Dict[str, Any]]) -> List:
        """Create atom featurizers from configuration."""
        featurizers = []

        for feat_config in atom_config:
            feat_type = feat_config["type"]
            oov = feat_config.get("oov", True)

            # Get vocabulary (custom or default)
            vocab = self.custom_vocab.get(feat_type, None)

            # Create featurizer
            if hasattr(atom, feat_type):
                featurizer_class = getattr(atom, feat_type)
                if vocab is not None:
                    # Convert list to set for vocab if needed
                    if isinstance(vocab, list):
                        vocab = set(vocab)
                    featurizer = featurizer_class(vocab=vocab, oov=oov)
                else:
                    featurizer = featurizer_class(oov=oov)
                featurizers.append(featurizer)
                logger.debug(
                    f"Created atom featurizer: {feat_type} (vocab size: {len(featurizer.vocab)})"
                )
            else:
                logger.warning(f"Unknown atom featurizer type: {feat_type}")

        return featurizers

    def _create_bond_featurizers(self, bond_config: List[Dict[str, Any]]) -> List:
        """Create bond featurizers from configuration."""
        featurizers = []

        for feat_config in bond_config:
            feat_type = feat_config["type"]
            oov = feat_config.get("oov", True)

            # Get vocabulary (custom or default)
            vocab = self.custom_vocab.get(feat_type, None)

            # Create featurizer
            if hasattr(bond, feat_type):
                featurizer_class = getattr(bond, feat_type)
                if vocab is not None:
                    # Convert list to set for vocab if needed
                    if isinstance(vocab, list):
                        vocab = set(vocab)
                    featurizer = featurizer_class(vocab=vocab, oov=oov)
                else:
                    featurizer = featurizer_class(oov=oov)
                featurizers.append(featurizer)
                logger.debug(
                    f"Created bond featurizer: {feat_type} (vocab size: {len(featurizer.vocab)})"
                )
            else:
                logger.warning(f"Unknown bond featurizer type: {feat_type}")

        return featurizers

    def get_featurizer_collection(self):
        """
        Get FeaturizerCollection with configured featurizers.

        Returns:
            FeaturizerCollection instance
        """
        if not GRAPHAMOLE_AVAILABLE:
            raise ImportError("Graphamole is not available")

        return collection.FeaturizerCollection(
            atom_featurizers=self.atom_featurizers,
            bond_featurizers=self.bond_featurizers,
        )

    def save_config_template(self, output_path: Path) -> None:
        """
        Save a template configuration file with default vocabularies.

        Args:
            output_path: Path where to save the template
        """
        # Collect all default vocabularies
        all_vocabs = {}

        # Add atom vocabularies
        for vocab_name, vocab_set in ATOM_DEFAULT_VOCAB.items():
            if isinstance(vocab_set, set):
                all_vocabs[vocab_name] = sorted(list(vocab_set))
            else:
                all_vocabs[vocab_name] = list(vocab_set)

        # Add bond vocabularies
        for vocab_name, vocab_set in BOND_DEFAULT_VOCAB.items():
            if isinstance(vocab_set, set):
                all_vocabs[vocab_name] = sorted(list(vocab_set))
            else:
                all_vocabs[vocab_name] = list(vocab_set)

        # Create template config
        template = {
            "vocabularies": all_vocabs,
            "atom_features": [
                {"type": "AtomType", "oov": True},
                {"type": "Hybridization", "oov": True},
                {"type": "CIPCode", "oov": True},
                {"type": "FormalCharge", "oov": True},
                {"type": "TotalNumHs", "oov": True},
                {"type": "TotalValence", "oov": True},
                {"type": "NumRadicalElectrons", "oov": True},
                {"type": "Degree", "oov": True},
                {"type": "RingSize", "oov": True},
                {"type": "ChiralCenter", "oov": True},
                {"type": "Aromatic", "oov": True},
                {"type": "Hetero", "oov": True},
                {"type": "HydrogenDonor", "oov": True},
                {"type": "HydrogenAcceptor", "oov": True},
                {"type": "Ring", "oov": True},
            ],
            "bond_features": [
                {"type": "BondType", "oov": True},
                {"type": "Stereo", "oov": True},
                {"type": "Conjugated", "oov": True},
                {"type": "Rotatable", "oov": True},
            ],
        }

        with open(output_path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(f"Feature configuration template saved to {output_path}")

    def get_vocab_summary(self) -> Dict[str, int]:
        """
        Get summary of vocabulary sizes for all features.

        Returns:
            Dictionary mapping feature names to vocabulary sizes
        """
        summary = {}

        # Atom feature vocab sizes
        for featurizer in self.atom_featurizers:
            feat_name = featurizer.__class__.__name__
            summary[f"atom_{feat_name}"] = len(featurizer.vocab)

        # Bond feature vocab sizes
        for featurizer in self.bond_featurizers:
            feat_name = featurizer.__class__.__name__
            summary[f"bond_{feat_name}"] = len(featurizer.vocab)

        return summary


def create_feature_config_template(output_path: Union[str, Path]) -> None:
    """
    Create a template feature configuration file.

    Args:
        output_path: Path where to save the template
    """
    manager = FeatureConfigManager()
    manager.save_config_template(Path(output_path))


def load_feature_config(config_path: Union[str, Path]):
    """
    Load feature configuration and return FeaturizerCollection.

    Args:
        config_path: Path to feature configuration file

    Returns:
        FeaturizerCollection with custom configuration
    """
    manager = FeatureConfigManager(Path(config_path))
    return manager.get_featurizer_collection()
