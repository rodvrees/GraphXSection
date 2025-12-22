"""Data extraction functions for GraphXSection_torch - PyTorch implementation."""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, List, Optional
from graphamole.featurizers import ALL_FEATURIZERS
from graphamole.featurizers.collection import FeaturizerCollection
from torch_geometric.data import Data, Batch
from rdkit import Chem


logger = logging.getLogger(__name__)


class AdductEncoder:
    """
    Robust adduct encoder that ensures consistent one-hot encoding across datasets.

    This class maintains a fixed vocabulary of adducts and their ordering, ensuring
    that train/validation/test splits and transfer learning scenarios use identical
    feature representations.

    Usage Examples:
    ---------------

    Basic usage with default vocabulary:
    >>> encoder = AdductEncoder()
    >>> encoder.fit([train_df['Adduct'], valid_df['Adduct'], test_df['Adduct']])
    >>> train_ohe = encoder.transform(train_df['Adduct'])
    >>> valid_ohe = encoder.transform(valid_df['Adduct'])

    Using custom vocabulary:
    >>> custom_vocab = ["(+H)", "(-H)", "(+Na)", "(+K)"]
    >>> encoder = AdductEncoder(adduct_vocabulary=custom_vocab)
    >>> encoder.fit([train_df['Adduct']])
    >>> train_ohe = encoder.transform(train_df['Adduct'])

    Transfer learning scenario - save and load vocabulary:
    >>> # During initial training
    >>> encoder = AdductEncoder()
    >>> encoder.fit([train_df['Adduct']])
    >>> encoder.save_vocabulary('adduct_vocab.json')
    >>>
    >>> # Later, for transfer learning or inference
    >>> encoder = AdductEncoder.load_vocabulary('adduct_vocab.json')
    >>> new_data_ohe = encoder.transform(new_df['Adduct'])

    Handling unknown adducts:
    >>> # Strict mode - raises error for unknown adducts
    >>> encoder.transform(data['Adduct'], handle_unknown='error')
    >>>
    >>> # Ignore mode - encodes unknown as zero vectors (default)
    >>> encoder.transform(data['Adduct'], handle_unknown='ignore')

    Key Features:
    -------------
    - Consistent feature ordering across all datasets
    - Handles missing adducts by creating zero vectors
    - Supports custom vocabularies for domain-specific data
    - Vocabulary persistence for reproducible experiments
    - Warning system for unknown adducts in new data
    - Efficient encoding using pre-computed index mappings

    Notes:
    ------
    - The order of adducts in the vocabulary determines the feature order
    - Unknown adducts are handled gracefully (no crash, warning logged)
    - All datasets encoded with the same encoder will have identical feature dimensions
    - The default vocabulary covers common MS adducts but can be customized
    """

    def __init__(self, adduct_vocabulary=None):
        """
        Initialize the adduct encoder.

        Args:
            adduct_vocabulary (List[str], optional): Custom list of adducts in desired order.
                If None, uses a comprehensive default vocabulary covering common MS adducts.

                Examples:
                - For ESI-positive only: ["(+H)", "(+Na)", "(+K)", "(+NH4)"]
                - For ESI-negative only: ["(-H)", "(-2H)", "(+HCOOH-H)"]
                - For minimal setup: ["(+H)", "(-H)"]
                - For comprehensive: Use None (default) to get all common adducts

                The order matters! Features will be generated in this exact order,
                ensuring consistency across all datasets and experiments.

        Note:
            Custom vocabularies should include all adducts you expect to encounter
            across training, validation, test sets, and future transfer learning tasks.
            Missing adducts from vocabulary will be encoded as zero vectors with warnings.
        """
        if adduct_vocabulary is None:
            # Default vocabulary based on common adducts in CCS data
            # Order is important for consistency across datasets
            self.adduct_vocabulary = [
                "(+H)",  # Most common positive mode adduct
                "(-H)",  # Most common negative mode adduct
                "(+Na)",  # Sodium adduct
                "(+K)",  # Potassium adduct
                "(+HCOOH-H)",  # Formic acid adduct,
                "(+K)",  # Potassium adduct
                # "(+Li)",  # Lithium adduct
                # "(+GirardP)",  # Girard P reagent
                # "(-e)",  # Electron capture
                # "(-HO)",  # Hydroxyl adduct
                # "(+TMT)",  # Tandem mass tag
                # "(+NH4)",  # Ammonium adduct
                # "(+Cl)",  # Chloride adduct,
                # "(-Cl)",  # Chloride loss
                # "(-CH3COO)",  # Acetate adduct
                # "(-CO+H)",  # Carbonyl loss
                # "(+2H)",  # Doubly charged proton
                # "(-2H)",  # Doubly charged deprotonation
                # "(+ACN+H)",  # Acetonitrile adduct
                # "(+CH3OH+H)",  # Methanol adduct
            ]
        else:
            self.adduct_vocabulary = list(adduct_vocabulary)

        # Create mapping from adduct to index for efficiency
        self.adduct_to_idx = {
            adduct: idx for idx, adduct in enumerate(self.adduct_vocabulary)
        }

        logger.info(
            f"AdductEncoder initialized with vocabulary: {self.adduct_vocabulary}"
        )

    def fit(self, adduct_series_list):
        """
        Fit the encoder to discover all adducts present in the data.

        This method analyzes all provided datasets to check for unknown adducts
        and logs warnings about vocabulary mismatches. It's recommended to call
        this once with ALL datasets (train/valid/test) before any transformations.

        Args:
            adduct_series_list (List[pd.Series]): List of pandas Series containing adduct values.
                Pass all your datasets at once for complete analysis:
                [train_df['Adduct'], valid_df['Adduct'], test_df['Adduct']]

        Returns:
            self: Returns self for method chaining (sklearn-style API)

        Example:
            >>> encoder = AdductEncoder()
            >>> encoder.fit([train_df['Adduct'], valid_df['Adduct'], test_df['Adduct']])
            >>> # Now encoder knows about all adducts across all splits
        """
        # Collect all unique adducts from all datasets
        all_adducts = set()
        for series in adduct_series_list:
            unique_adducts = set(series.dropna().unique())
            all_adducts.update(unique_adducts)

        # Check for unknown adducts
        unknown_adducts = all_adducts - set(self.adduct_vocabulary)
        if unknown_adducts:
            logger.warning(
                f"Found unknown adducts not in vocabulary: {unknown_adducts}"
            )
            logger.warning(
                "Consider updating the adduct vocabulary to include these adducts"
            )

        # Check for missing adducts (vocabulary items not in data)
        missing_adducts = set(self.adduct_vocabulary) - all_adducts
        if missing_adducts:
            logger.info(f"Vocabulary adducts not present in data: {missing_adducts}")

        return self

    def _encode_one(self, adduct, handle_unknown: str = "ignore") -> np.ndarray:
        """Encode a single adduct into a 1D one-hot vector."""
        n_adducts = len(self.adduct_vocabulary)
        vec = np.zeros((n_adducts,), dtype=float)
        if (
            adduct is None
            or (isinstance(adduct, float) and np.isnan(adduct))
            or pd.isna(adduct)
        ):
            return vec
        a = str(adduct)
        if a in self.adduct_to_idx:
            vec[self.adduct_to_idx[a]] = 1.0
            return vec
        if handle_unknown == "error":
            raise ValueError(
                f"Unknown adduct: {a}. Known adducts: {self.adduct_vocabulary}"
            )
        logger.warning("Unknown adduct '%s', encoding as zero vector", a)
        return vec

    def transform(self, adducts, handle_unknown: str = "ignore"):
        """
        Transform adduct(s) to one-hot.

        Accepts a single adduct string (returns shape (n_vocab,)) or a sequence/
        pandas Series of adduct strings (returns shape (n_samples, n_vocab)).
        Unknowns are zero vectors unless handle_unknown='error'.
        """
        # Sequence input -> (N, V)
        if isinstance(adducts, (list, tuple, np.ndarray, pd.Series)):
            adduct_list = list(adducts)
            if len(adduct_list) == 0:
                return np.zeros((0, len(self.adduct_vocabulary)), dtype=float)
            return np.stack(
                [self._encode_one(a, handle_unknown) for a in adduct_list], axis=0
            )
        # Scalar input -> (V,)
        return self._encode_one(adducts, handle_unknown)

    def fit_transform(self, adduct_series_list, handle_unknown="ignore"):
        """
        Fit the encoder and transform the first series in one step.

        Args:
            adduct_series_list: List of pandas Series, first one will be transformed
            handle_unknown: How to handle unknown adducts

        Returns:
            np.ndarray: One-hot encoded adducts for the first series
        """
        self.fit(adduct_series_list)
        return self.transform(adduct_series_list[0], handle_unknown)

    def get_feature_names(self):
        """
        Get feature names for the one-hot encoded columns.

        Returns:
            List[str]: Feature names in format 'Adduct_{adduct_name}'
        """
        return [f"Adduct_{adduct}" for adduct in self.adduct_vocabulary]

    def save_vocabulary(self, filepath):
        """Save the adduct vocabulary to a file for later use."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.adduct_vocabulary, f, indent=2)
        logger.info(f"Adduct vocabulary saved to {filepath}")

    @classmethod
    def load_vocabulary(cls, filepath):
        """Load adduct vocabulary from a file."""
        import json

        with open(filepath, "r") as f:
            vocabulary = json.load(f)
        logger.info(f"Adduct vocabulary loaded from {filepath}")
        return cls(adduct_vocabulary=vocabulary)


# Example: Complete workflow with AdductEncoder
"""
Comprehensive AdductEncoder Workflow Example:

# 1. Basic usage with default vocabulary
import pandas as pd
from GraphXSection_torch.data_extract import AdductEncoder

# Load your data
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')

# Create encoder and fit to all data splits
encoder = AdductEncoder()
encoder.fit([train_df['Adduct'], valid_df['Adduct'], test_df['Adduct']])

# Transform each dataset (guaranteed consistent dimensions)
train_ohe = encoder.transform(train_df['Adduct'])
valid_ohe = encoder.transform(valid_df['Adduct'])
test_ohe = encoder.transform(test_df['Adduct'])

print(f"All datasets have {train_ohe.shape[1]} features")
print(f"Feature names: {encoder.get_feature_names()}")

# 2. Custom vocabulary for specific MS conditions
esi_pos_vocab = ["(+H)", "(+Na)", "(+K)", "(+NH4)", "(+2H)"]
encoder_esi_pos = AdductEncoder(adduct_vocabulary=esi_pos_vocab)
encoder_esi_pos.fit([train_df['Adduct']])
train_ohe_custom = encoder_esi_pos.transform(train_df['Adduct'])

# 3. Transfer learning scenario
# Save vocabulary during initial training
encoder.save_vocabulary('my_adduct_vocab.json')

# Later, load for new experiments or deployment
encoder_loaded = AdductEncoder.load_vocabulary('my_adduct_vocab.json')
new_data_ohe = encoder_loaded.transform(new_df['Adduct'])

# 4. Handling different data distributions
# Train set has: (+H), (-H), (+Na)
# Test set has: (+H), (-H), (+K)  # (+K) is unknown, (+Na) is missing
# Result: (+K) becomes zero vector, (+Na) feature still exists (all zeros)

# 5. Strict mode for development
try:
    encoder.transform(problematic_df['Adduct'], handle_unknown='error')
except ValueError as e:
    print(f"Found unknown adducts: {e}")
    # Update vocabulary or clean data
"""


def _stratified_split(
    df: pd.DataFrame,
    test_split: float = 0.1,
    valid_split: float = 0.1,
    stratify_col: str = "Adduct",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified split of the datafram into training, validation, and test sets.

    Args:
        df (pd.DataFrame): Dataframe to split.
        test_split (float): Fraction of the data to use for testing.
        valid_split (float): Fraction of the data to use for validation.
        stratify_col (str): Column to stratify the data on.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Dataframes for training, validation, and testing.
    """

    # Initialize empty dataframes for train, valid, and test
    df_train = pd.DataFrame(columns=df.columns)
    df_valid = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)

    # Group by the stratify column
    grouped = df.groupby(stratify_col)

    for _, group in grouped:
        # Split the group into train and test
        train, test = train_test_split(group, test_size=test_split, random_state=42)
        # Split the train group into train and valid
        train, valid = train_test_split(train, test_size=valid_split, random_state=42)

        df_train = pd.concat([df_train, train])
        df_valid = pd.concat([df_valid, valid])
        df_test = pd.concat([df_test, test])

    return df_train, df_valid, df_test


class MoleculeIon:
    """Molecule object for featurization and data extraction."""

    def __init__(self, input_smiles: str | Chem.Mol, adduct: str = None):
        """
        Molecule object for featurization and data extraction.

        Parameters
        ----------
        input_smiles
            SMILES string or RDKit molecule object.

        """
        self.input_smiles = input_smiles
        self.adduct = adduct

    def get_mol(self) -> Chem.Mol:
        """Return an RDKit molecule object."""
        # TODO: Implement molecule sanitizing?
        if isinstance(self.input_smiles, str):
            return Chem.MolFromSmiles(self.input_smiles)
        elif isinstance(self.input_smiles, Chem.Mol):
            return self.input_smiles
        else:
            raise ValueError(f"Invalid input representation: {self.input_smiles}")

    def get_data(
        self,
        featurizer_collection: FeaturizerCollection | None = None,
        one_hot_encode: bool = False,
    ) -> Data:
        """
        Return a :py:class:`torch_geometric.data.Data` object for the molecule.

        Parameters
        ----------
        featurizer_collection
            Collection of featurizers for atoms and bonds. Uses all featurizers by default.

        """
        if featurizer_collection is None:
            featurizer_collection = ALL_FEATURIZERS

        mol = self.get_mol()

        atom_features_list: list[list[int]] = []
        edge_indices: list[list[int]] = []
        edge_features_list: list[list[int]] = []

        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_features_list.append(
                featurizer_collection.get_atom_features(
                    atom, one_hot_encode=one_hot_encode
                )
            )

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # Include both directions.
            bond_features = featurizer_collection.get_bond_features(bond)
            edge_features_list.extend([bond_features, bond_features])

        # Use float features so downstream MLPs can learn meaningful combinations
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        # Return only the molecular graph; global/adduct features are handled by the dataset
        # to ensure consistency with the fitted AdductEncoder.
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GraphXSectionDataset(Dataset):
    """Graph dataset with adduct/global features, MoleculeDataset-style.

    Each item: {
        'graph': PyG Data (x: float32, edge_attr: float32, edge_index: long),
        'global_feats': float32 [F] (e.g., adduct OHE),
        'target': float32 [1]
    }

    Supports optional virtual node representation for adducts.
    """

    def __init__(
        self,
        smiles_list: List[str],
        adduct_list: List[str],
        global_feats: np.ndarray,
        targets: np.ndarray,
        featurizer_collection: FeaturizerCollection | None = None,
        one_hot_encode: bool = False,
        feature_config_path: Path | None = None,
    ):
        """
        Store raw inputs; graphs are featurized on demand in get().

        Args:
            smiles_list: List of SMILES strings.
            adduct_list: List of adduct strings aligned with smiles_list.
            global_feats: Array of global features (e.g., adduct OHE), shape (N, F).
            targets: Array of targets, shape (N,) or (N,1).
            featurizer_collection: Custom featurizer collection.
            one_hot_encode: Whether to use one-hot encoding for features.
            feature_config_path: Path to custom feature configuration JSON.
        """
        super().__init__()
        if not (
            len(smiles_list) == len(adduct_list) == len(global_feats) == len(targets)
        ):
            raise ValueError(
                "smiles, adducts, global_feats, and targets must have equal length"
            )

        # Setup featurizer collection with optional custom config
        if featurizer_collection is not None:
            self.featurizer_collection = featurizer_collection
        elif feature_config_path is not None:
            # Try to load custom feature configuration
            try:
                from GraphXSection_torch.feature_config import FeatureConfigManager

                feature_manager = FeatureConfigManager(feature_config_path)
                self.featurizer_collection = feature_manager.get_featurizer_collection()
                logger.info(f"Loaded custom featurizers from {feature_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load feature config: {e}")
                logger.warning("Falling back to default featurizers")
                self.featurizer_collection = ALL_FEATURIZERS
        else:
            self.featurizer_collection = ALL_FEATURIZERS

        self._molecule_list = self._load_molecules(smiles_list, adduct_list)
        self.global_feats = torch.as_tensor(global_feats, dtype=torch.float32)
        t = torch.as_tensor(targets, dtype=torch.float32)
        self.targets = t.view(-1, 1) if t.ndim == 1 else t
        self.one_hot_encode = one_hot_encode

    def __len__(self) -> int:
        return len(self._molecule_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get(idx)

    @staticmethod
    def _load_molecules(
        smiles_list: List[str], adduct_list: List[str]
    ) -> List[MoleculeIon]:
        return [MoleculeIon(smi, add) for smi, add in zip(smiles_list, adduct_list)]

    def get(self, idx: int) -> Dict[str, Any]:
        graph: Data = self._molecule_list[idx].get_data(
            self.featurizer_collection, one_hot_encode=self.one_hot_encode
        )
        return {
            "graph": graph,
            "global_feats": self.global_feats[idx],
            "target": self.targets[idx],
        }

    def summarize(self):
        """Print a summary of the dataset."""
        logger.info("--------------Dataset Summary--------------")
        logger.info(f"Dataset size: {len(self)} samples")
        logger.info(f"Global features shape: {self.global_feats.shape}")
        logger.info(f"Targets shape: {self.targets.shape}")
        logger.info("-----------------------------------------")
        first = self[0]
        # Print smiles if available
        if isinstance(self._molecule_list[0].input_smiles, str):
            logger.info(f"First SMILES: {self._molecule_list[0].input_smiles}")
        else:
            logger.info("First SMILES: Not available")

        # Print adduct if available
        if self._molecule_list[0].adduct is not None:
            logger.info(f"First Adduct: {self._molecule_list[0].adduct}")
        else:
            logger.info("First Adduct: Not available")
        logger.info("-----------------------------------------")
        logger.info(
            f"First graph nodes: {first['graph'].num_nodes}, edges: {first['graph'].num_edges}"
        )
        logger.info(f"First graph node features shape: {first['graph'].x.shape}")
        logger.info(f"First graph node features: {first['graph'].x}")
        ea = getattr(first["graph"], "edge_attr", None)
        logger.info(
            f"First graph edge features shape: {tuple(ea.shape) if ea is not None else None}"
        )
        logger.info(f"First graph edge features: {first['graph'].edge_attr}")
        logger.info(f"First global features shape: {first['global_feats'].shape}")
        logger.info(f"First global features: {first['global_feats']}")
        logger.info(f"First target: {first['target']}")
        logger.info("-----------------------------------------")


def collate_fn(batch):
    """Batch dicts of graphs, global features, and targets."""
    graphs = [item["graph"] for item in batch]
    global_feats = torch.stack([item["global_feats"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)

    batched_graphs = Batch.from_data_list(graphs)
    return {"graph": batched_graphs, "global_feats": global_feats, "target": targets}


def _neutral_exact_mass_from_smiles(smiles: str) -> float:
    """Calculate the neutral exact mass of a molecule from its SMILES string.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        Neutral exact molecular weight as a float
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return float(rdMolDescriptors.CalcExactMolWt(mol))


def _get_data(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    train: bool = False,
    adduct_encoder: AdductEncoder | None = None,
    use_adduct_descriptors: bool = True,
) -> GraphXSectionDataset:
    """Extract data necessary for training from the dataframe and build the dataset.

    Args:
        df: DataFrame with molecular data
        smiles_col: Column name for SMILES strings
        train: Whether this is training data (currently unused but kept for compatibility)
        adduct_encoder: Legacy AdductEncoder (only used when use_adduct_descriptors=False)
        use_adduct_descriptors: Whether to use physicochemical descriptors (True, default)
                               or legacy one-hot encoding (False)

    Returns:
        GraphXSectionDataset with extracted features
    """

    # Remove rows with NaN values in the SMILES column or the global features
    df = df.dropna(
        subset=[
            smiles_col,
            "Adduct",
            "CCS",
        ]
    )
    smiles = df[smiles_col].values.tolist()
    adducts = df["Adduct"].values.tolist()

    # Choose between descriptor-based or one-hot encoding approach
    if use_adduct_descriptors:
        # Extract physicochemical descriptors for each adduct
        from GraphXSection_torch.adduct_features import AdductFeatureExtractor

        logger.info("Extracting physicochemical adduct descriptors (vocab-free)")
        extractor = AdductFeatureExtractor()

        # Extract descriptors for each adduct (includes SMILES information)
        all_feats = np.stack(
            [
                extractor.extract_features(adduct, smiles=smi)
                for adduct, smi in zip(adducts, smiles)
            ]
        ).astype(np.float32)

        logger.info(f"Extracted {all_feats.shape[1]}-dimensional descriptors")
    else:
        # Use AdductEncoder if provided, otherwise create a temporary one on this split
        if adduct_encoder is not None:
            adduct_OHE = adduct_encoder.transform(df["Adduct"])
        else:
            logger.warning(
                "No AdductEncoder provided; fitting a temporary encoder on this dataframe"
            )
            _tmp_encoder = AdductEncoder().fit([df["Adduct"]])
            adduct_OHE = _tmp_encoder.transform(df["Adduct"])

        all_feats = adduct_OHE

    logger.info("df.columns: %s", df.columns.to_list())
    logger.info(f"all_feats shape: {all_feats.shape}")

    y = df["CCS"].values

    logger.info(f"Data shape: {len(smiles)}, {all_feats.shape}, {y.shape}")

    dataset = GraphXSectionDataset(smiles, adducts, all_feats, y)

    logger.info(f"Created dataset with {len(dataset)} samples")
    logger.info(
        f"First graph nodes: {dataset[0]['graph'].num_nodes}, edges: {dataset[0]['graph'].num_edges}"
    )
    logger.info(f"Global feats shape: {dataset[0]['global_feats'].shape}")

    return dataset


def get_embeddings(mol_df, config):
    """
    Get the embeddings for the molecules using a pretrained model.

    Note: This is a placeholder - you'll need to implement the actual
    pretrained model loading and inference for PyTorch.

    Args:
        mol_df (pd.DataFrame): Dataframe with the molecules.
        config (Dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Dataframe with embeddings added.
    """
    # TODO: Implement pretrained model loading and embedding generation
    # For now, create dummy embeddings
    logger.warning("Using dummy embeddings - implement actual pretrained model!")

    smiles = mol_df[config["smiles_col"]].tolist()
    # Create dummy embeddings of size 512 (matching expected size)
    dummy_embeddings = np.random.randn(len(smiles), 512)
    mol_df["embeddings"] = dummy_embeddings.tolist()

    return mol_df


def data_extract(config: Dict[str, Any]) -> Tuple[
    GraphXSectionDataset,
    GraphXSectionDataset,
    GraphXSectionDataset,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Extract the data for training the model based on the configuration provided.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[GraphXSectionDataset, GraphXSectionDataset, GraphXSectionDataset, pd.DataFrame, pd.DataFrame]:
        Training, validation, and test datasets, the validation and test dataframes.
    """

    logger.info("Reading data file: %s", config["data_path"])
    data = pd.read_csv(config["data_path"])

    # if config["model"].get("type", None) == "Ginny3D":
    #     data = pd.read_csv(config["data_path"])
    #     data.drop(["delta_to_second_best"], axis=1, inplace=True)
    #     # Remove columns where all values are the same
    #     data.replace(-1000, np.nan, inplace=True)
    #     data = _drop_constant_columns(data)
    #     # Drop columns with NaN values
    #     logger.info("Dropping columns with NaN values")
    #     logger.debug(
    #         "Columns with NaN values: %s", len(data.columns[data.isna().any()])
    #     )
    #     data.dropna(axis=1, inplace=True)
    #     logger.info("Dropped columns with NaN values")
    #     logger.debug("Remaining columns: %s", len(data.columns))

    df_train, df_valid, df_test = _stratified_split(
        data,
        test_split=config["test_split"],
        valid_split=config["valid_split"],
        stratify_col="Adduct",
    )

    # if config["model"].get("type", None) == "Eddy":
    #     df_train = get_embeddings(df_train, config)
    #     df_valid = get_embeddings(df_valid, config)
    #     df_test = get_embeddings(df_test, config)
    #     df_train["Adduct_OHE"] = _get_adduct_OHE(df_train).tolist()
    #     df_valid["Adduct_OHE"] = _get_adduct_OHE(df_valid).tolist()
    #     df_test["Adduct_OHE"] = _get_adduct_OHE(df_test).tolist()

    #     return df_train, df_valid, df_test, None, None

    if config.get("save_dfs", False):
        df_train.to_csv(f"{config['output_path']}/train_df.csv", index=False)
        df_valid.to_csv(f"{config['output_path']}/valid_df.csv", index=False)
        df_test.to_csv(f"{config['output_path']}/test_df.csv", index=False)

    logger.debug(
        f"Training data shape: {df_train.shape}, {df_valid.shape}, {df_test.shape}"
    )

    # Check if using adduct descriptors (recommended for generalization)
    use_adduct_descriptors = config.get("use_adduct_descriptors", True)

    if use_adduct_descriptors:
        logger.info(
            "Using physicochemical adduct descriptors for generalization (vocab-free)"
        )
        adduct_encoder = None
    else:
        # Create and fit AdductEncoder for consistent one-hot encoding across all splits
        logger.info(
            "Creating AdductEncoder for consistent adduct encoding (legacy mode)"
        )
        adduct_encoder = AdductEncoder()
        adduct_encoder.fit([df_train["Adduct"], df_valid["Adduct"], df_test["Adduct"]])

        # Log adduct information for debugging
        logger.info(f"Adduct encoder vocabulary: {adduct_encoder.adduct_vocabulary}")
        logger.info(f"Adduct feature names: {adduct_encoder.get_feature_names()}")

        # Save adduct vocabulary for transfer learning
        if config.get("save_dfs", False):
            vocab_path = f"{config['output_path']}/adduct_vocabulary.json"
            adduct_encoder.save_vocabulary(vocab_path)

    logger.info("Extracting training data")
    train_dataset = _get_data(
        df_train,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=True,
        adduct_encoder=adduct_encoder,
        use_adduct_descriptors=use_adduct_descriptors,
    )

    logger.info("Extracting validation and test data")
    valid_dataset = _get_data(
        df_valid,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=False,
        adduct_encoder=adduct_encoder,
        use_adduct_descriptors=use_adduct_descriptors,
    )
    test_dataset = _get_data(
        df_test,
        smiles_col=config.get("smiles_col", "SMILES"),
        train=False,
        adduct_encoder=adduct_encoder,
        use_adduct_descriptors=use_adduct_descriptors,
    )

    logger.info("Data extraction complete")

    return train_dataset, valid_dataset, test_dataset, df_test, df_valid
