"""Adduct feature encoding for virtual node representation."""

import numpy as np
import re
from typing import Dict, List


class AdductFeatureExtractor:
    """
    Extract structured features from adduct strings for virtual node initialization.

    Features extracted:
    - Charge magnitude (absolute value)
    - Charge sign (positive/negative)
    - Ion type (proton, sodium, potassium, formate, etc.)
    - Mass delta (approximate)

    Note: This implementation is a placeholder and can be refined later.
    """

    # Atomic masses (simplified)
    ATOMIC_MASSES = {
        "H": 1.008,
        "Na": 22.990,
        "K": 39.098,
        "NH4": 18.038,  # Ammonium
        "Li": 6.941,
        "Mg": 24.305,
        "Ca": 40.078,
        "Fe": 55.845,
        "Cl": 35.453,
        "Br": 79.904,
        "HCOO": 44.998,  # Formate
        "CH3COO": 59.044,  # Acetate
    }

    # Ion type one-hot encoding
    ION_TYPES = [
        "H",  # Proton
        "Na",  # Sodium
        "K",  # Potassium
        "NH4",  # Ammonium
        "Li",  # Lithium
        "Metal",  # Other metals (Mg, Ca, Fe, etc.)
        "HCOO",  # Formate
        "Halide",  # Cl, Br
        "Other",  # Other/unknown
    ]

    def __init__(self, feature_dim: int = 16):
        """
        Initialize adduct feature extractor.

        Parameters
        ----------
        feature_dim : int
            Target dimension for adduct features (will pad/truncate).
        """
        self.feature_dim = feature_dim

    def parse_adduct(self, adduct_str: str) -> Dict[str, any]:
        """
        Parse adduct string to extract charge and ion information.

        Examples:
        - "(+H)" -> charge=+1, ions=['H']
        - "(-2H)" -> charge=-2, ions=['H']
        - "(+Na)" -> charge=+1, ions=['Na']
        - "(+2Na-H)" -> charge=+1, ions=['Na', 'Na', '-H']

        Parameters
        ----------
        adduct_str : str
            Adduct string (e.g., "(+H)", "(-2H)", "(+Na)")

        Returns
        -------
        dict
            Dictionary with 'charge' (int), 'ions' (list), 'mass_delta' (float)
        """
        # Remove parentheses and spaces
        adduct = adduct_str.strip("() ")

        # Determine overall charge sign
        charge_sign = 1 if adduct.startswith("+") else -1

        # Parse components
        # Pattern: optional number, optional +/-, ion name
        pattern = r"([+-]?)(\d*)([A-Za-z0-9]+)"
        matches = re.findall(pattern, adduct)

        total_charge = 0
        ions = []
        mass_delta = 0.0

        for sign_str, count_str, ion in matches:
            if not ion:
                continue

            count = int(count_str) if count_str else 1
            sign = -1 if sign_str == "-" else 1

            total_charge += sign * count
            ions.extend([ion] * count)

            # Calculate mass contribution
            if ion in self.ATOMIC_MASSES:
                mass_delta += sign * count * self.ATOMIC_MASSES[ion]

        # Do NOT multiply by charge_sign again - the regex already captures per-component signs
        return {
            "charge": total_charge,
            "ions": ions,
            "mass_delta": mass_delta,
        }

    def get_ion_type_encoding(self, ions: List[str]) -> np.ndarray:
        """
        Create one-hot encoding for ion types.

        Parameters
        ----------
        ions : List[str]
            List of ion symbols

        Returns
        -------
        np.ndarray
            One-hot encoded ion types (multi-hot if multiple ions)
        """
        encoding = np.zeros(len(self.ION_TYPES), dtype=np.float32)

        for ion in ions:
            if ion == "H":
                encoding[0] = 1
            elif ion == "Na":
                encoding[1] = 1
            elif ion == "K":
                encoding[2] = 1
            elif ion == "NH4":
                encoding[3] = 1
            elif ion == "Li":
                encoding[4] = 1
            elif ion in ["Mg", "Ca", "Fe", "Zn", "Cu"]:
                encoding[5] = 1
            elif "HCOO" in ion or "formate" in ion.lower():
                encoding[6] = 1
            elif ion in ["Cl", "Br", "I", "F"]:
                encoding[7] = 1
            else:
                encoding[8] = 1  # Other/unknown

        return encoding

    def extract_features(self, adduct_str: str) -> np.ndarray:
        """
        Extract full feature vector for adduct.

        Feature composition:
        - Charge magnitude (1 value, normalized)
        - Charge sign (2 values, one-hot: positive/negative)
        - Ion type (9 values, multi-hot)
        - Mass delta (1 value, normalized)
        - Padding to reach feature_dim

        Parameters
        ----------
        adduct_str : str
            Adduct string

        Returns
        -------
        np.ndarray
            Feature vector of shape (feature_dim,)
        """
        parsed = self.parse_adduct(adduct_str)

        # Charge magnitude (normalized by common range 1-3)
        charge_mag = np.array([abs(parsed["charge"]) / 3.0], dtype=np.float32)

        # Charge sign (one-hot)
        charge_sign = np.zeros(2, dtype=np.float32)
        if parsed["charge"] > 0:
            charge_sign[0] = 1.0  # Positive
        else:
            charge_sign[1] = 1.0  # Negative

        # Ion type encoding
        ion_encoding = self.get_ion_type_encoding(parsed["ions"])

        # Mass delta (normalized by common range 0-50 Da)
        mass_delta = np.array([parsed["mass_delta"] / 50.0], dtype=np.float32)

        # Concatenate features
        features = np.concatenate(
            [
                charge_mag,  # 1
                charge_sign,  # 2
                ion_encoding,  # 9
                mass_delta,  # 1
            ]
        )  # Total: 13 features

        # Pad or truncate to target dimension
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        elif len(features) > self.feature_dim:
            features = features[: self.feature_dim]

        # Keep continuous features as float32 for the adduct_encoder MLP
        return features.astype(np.float32)
