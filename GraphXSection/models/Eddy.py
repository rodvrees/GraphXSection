import tensorflow as tf
import logging
from molgraph import chemistry

logger = logging.getLogger(__name__)
SAVED_MODEL_PATH = (
    "/mnt/c/Users/robbe/Work/Waters_GNN_SmalMol/molecular-encodings/saved_model_PC"
)


def get_encoder():
    atom_encoder = chemistry.Tokenizer(
        [
            chemistry.features.Symbol(),
            chemistry.features.Hybridization(),
            chemistry.features.FormalCharge(),
            chemistry.features.TotalNumHs(),
            chemistry.features.TotalValence(),
            chemistry.features.Degree(),
            chemistry.features.Aromatic(),
            chemistry.features.Hetero(),
            chemistry.features.HydrogenDonor(),
            chemistry.features.HydrogenAcceptor(),
            chemistry.features.Ring(),
        ]
    )

    bond_encoder = chemistry.Tokenizer(
        [
            chemistry.features.BondType(),
            chemistry.features.Conjugated(),
            chemistry.features.Rotatable(),
        ]
    )

    return chemistry.MolecularGraphEncoder(
        atom_encoder,
        bond_encoder,
        positional_encoding_dim=None,
    )


class PretrainedModel:
    def __init__(self, model):
        self._model = model
        self._encoder = get_encoder()

    @classmethod
    def from_path(cls, path):
        return cls(tf.saved_model.load(path))

    def __call__(self, smiles):
        encoded = self._encoder(smiles)
        return self._model(encoded).numpy()


def get_embeddings(mol_df, config):
    """
    Get the embeddings for the molecules.

    Args:
        mol_df (pd.DataFrame): Dataframe with the molecules.

    Returns:
        np.ndarray: Embeddings for the molecules.
    """
    # Load the model
    model = PretrainedModel.from_path(SAVED_MODEL_PATH)
    # Get the embeddings
    smiles = mol_df[config["smiles_col"]].tolist()
    embeddings = model(smiles)
    mol_df["embeddings"] = embeddings.tolist()
    return mol_df


def Eddy(config):
    """
    Train the Eddy model.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        config (Dict[str, Any]): Training configuration.

    Returns:
        keras.Model: Trained model.
    """
    # Define the model
    logger.info("Loading Eddy model")

    # MLP for regression

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    return model
