"""Adduct feature encoding for virtual node representation."""

import numpy as np
import re
import warnings
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Physical constants for MR -> polarizability conversion
N_A = 6.02214076e23  # 1/mol
CM3_TO_A3 = 1e24  # 1 cm^3 = 1e24 Å^3

# For MS adduct mass delta, treat H as proton mass (not neutral H atom)
PROTON_MASS = 1.007276466812


class AdductFeatureExtractor:
    """
    Extract structured features from adduct strings for virtual node initialization.

    Features extracted:
    - Charge magnitude (absolute value)
    - Charge sign (positive/negative)
    - Mass delta (calculated from molecular formula composition)

    Mass calculation strategy:
    Parses molecular formulas (e.g., "HCOOH" -> H2CO2) and calculates monoisotopic
    mass from elemental composition. Falls back to hardcoded values only for
    components that cannot be parsed as formulas (e.g., "e" for electron, "NH4").
    """

    # Exact atomic masses for elements
    ELEMENT_MASSES = {
        "Ag": 106.905093,
        "Al": 26.981538,
        "Ar": 39.962383,
        "As": 74.921595,
        "Au": 196.966560,
        "B": 11.009305,
        "Ba": 137.905236,
        "Be": 9.012183,
        "Bi": 208.980388,
        "Br": 78.918336,
        "C": 12.000000,
        "H": 1.007825,
        "Hg": 201.970632,
        "C": 12.000000,
        "Ca": 39.962591,
        "Ce": 139.905442,
        "Cs": 132.905433,
        "Cl": 34.968853,
        "Cr": 51.940510,
        "Co": 58.933198,
        "Cu": 62.929599,
        "Dy": 163.929183,
        "Er": 165.930305,
        "Eu": 152.921243,
        "F": 18.998403,
        "Fe": 55.934939,
        "Ga": 68.925581,
        "Gd": 157.924111,
        "Ge": 73.921179,
        "H": 1.007825,
        "Hf": 179.946561,
        "He": 4.002603,
        "Ho": 164.930332,
        "I": 126.904477,
        "In": 114.903875,
        "Ir": 192.962942,
        "K": 38.963708,
        "Kr": 83.911506,
        "La": 138.906355,
        "Li": 7.016005,
        "Lu": 174.940785,
        "Mg": 23.985045,
        "Mn": 54.938046,
        "Mo": 97.905405,
        "N": 14.003074,
        "Na": 22.989770,
        "Nd": 141.907731,
        "Ne": 19.992439,
        "Ni": 57.935347,
        "Nb": 92.906378,
        "O": 15.994915,
        "Os": 191.961487,
        "P": 30.973763,
        "Pb": 207.976641,
        "Pd": 105.903475,
        "Pt": 194.964774,
        "Pr": 140.907657,
        "Rb": 84.911800,
        "Re": 186.955765,
        "Rh": 102.905503,
        "Ru": 101.904348,
        "S": 31.972072,
        "Sb": 120.903824,
        "Sc": 44.955914,
        "Se": 79.916521,
        "Si": 27.976928,
        "Sm": 151.919741,
        "Sn": 117.902199,
        "Sr": 86.905625,
        "Ta": 180.948014,
        "Tb": 158.925350,
        "Te": 129.906229,
        "Ti": 47.947947,
        "Th": 232.038054,
        "Tl": 204.974410,
        "Tm": 168.934225,
        "U": 238.050786,
        "V": 50.943963,
        "W": 183.950935,
        "Xe": 131.904148,
        "Y": 88.905856,
        "Yb": 173.938873,
        "Zn": 63.929145,
        "Zr": 89.904708,
    }

    # Fallback for components that are not chemical formulas
    SIMPLE_ION_MASSES = {
        "e": 0.00054858,  # Electron
        "NH4": 18.033823,  # Ammonium
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

    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """
        Parse a molecular formula string into element counts.

        Examples:
        - "H" -> {"H": 1}
        - "H2O" -> {"H": 2, "O": 1}
        - "HCOOH" -> {"H": 2, "C": 1, "O": 2}
        - "CH3COO" -> {"C": 2, "H": 3, "O": 2}

        Parameters
        ----------
        formula : str
            Molecular formula (e.g., "H2O", "HCOOH", "CH3COO")

        Returns
        -------
        Dict[str, int]
            Dictionary mapping element symbols to counts
        """
        # Pattern matches element symbol (uppercase + optional lowercase) followed by optional number
        pattern = r"([A-Z][a-z]?)(\d*)"
        matches = re.findall(pattern, formula)

        composition = {}
        for element, count_str in matches:
            if element:  # Ignore empty matches
                count = int(count_str) if count_str else 1
                composition[element] = composition.get(element, 0) + count

        return composition

    def _calculate_mass_from_formula(self, formula: str) -> Optional[float]:
        """
        Calculate monoisotopic mass from molecular formula composition.

        Examples:
        - "H" -> 1.007825
        - "Na" -> 22.989769
        - "HCOOH" -> 46.00550 (H: 2x1.008 + C: 12.000 + O: 2x15.995)
        - "CH3COO" -> 59.01385

        Parameters
        ----------
        formula : str
            Molecular formula string

        Returns
        -------
        Optional[float]
            Monoisotopic mass in Daltons, or None if formula cannot be parsed
        """
        try:
            composition = self._parse_formula(formula)

            if not composition:
                return None

            mass = 0.0
            for element, count in composition.items():
                if element not in self.ELEMENT_MASSES:
                    warnings.warn(
                        f"Unknown element '{element}' in formula '{formula}'. "
                        f"Mass calculation may be inaccurate.",
                        UserWarning,
                    )
                    return None
                mass += self.ELEMENT_MASSES[element] * count

            return mass

        except Exception as e:
            warnings.warn(f"Failed to parse formula '{formula}': {e}", UserWarning)
            return None

    def _get_component_mass(self, component: str) -> float:
        """
        Get the mass of an adduct component.

        Strategy:
        1. Try to parse as molecular formula and calculate mass
        2. Fall back to hardcoded simple ion masses
        3. Warn and return 0 if component is unknown

        Parameters
        ----------
        component : str
            Component identifier (e.g., "H", "Na", "HCOOH", "CH3COO")

        Returns
        -------
        float
            Mass in Daltons
        """
        # First try formula-based calculation
        mass = self._calculate_mass_from_formula(component)
        if mass is not None:
            return mass

        # Fall back to simple ions
        if component in self.SIMPLE_ION_MASSES:
            return self.SIMPLE_ION_MASSES[component]

        # Unknown component
        warnings.warn(
            f"Unknown component '{component}'. Cannot determine mass. Using 0.0",
            UserWarning,
        )
        return 0.0

    def parse_adduct(self, adduct_str: str) -> Dict[str, any]:
        """
        Parse adduct string to extract charge and composition information.

        Handles multiple notation formats:
        1. Explicit charge: [nM + Adduct +/- Loss]z+/- (e.g., "[M+H]+", "[2M-H]2-")
        2. Composition-based: (+/-Components) (e.g., "(+H)", "(+HCOOH-H)", "(-2H)")

        The key principle: charge is parsed separately from composition when explicitly
        specified. Otherwise, charge is inferred from signed component counts as a
        fallback heuristic.

        Examples:
        - "(+H)" -> charge=+1, composition=[('+', 1, 'H')], mass_delta=+1.008
        - "(-2H)" -> charge=-2, composition=[('-', 2, 'H')], mass_delta=-2.016
        - "(+HCOOH-H)" -> charge=-1, composition=[('+', 1, 'HCOOH'), ('-', 1, 'H')], mass_delta=+44.998
        - "[M+H]+" -> charge=+1, composition=[('+', 1, 'H')]
        - "[2M+Na]+" -> charge=+1, composition=[('+', 1, 'Na')], multiplier=2
        - "[M+HCOO]-" -> charge=-1, composition=[('+', 1, 'HCOO')], mass_delta=+44.998

        Parameters
        ----------
        adduct_str : str
            Adduct string in various formats

        Returns
        -------
        dict
            Dictionary with:
            - 'charge' (int): Explicit charge state or composition-derived fallback
            - 'composition' (list): List of (sign, count, component) tuples
            - 'mass_delta' (float): Total mass shift in Daltons
            - 'charge_explicit' (bool): Whether charge was explicitly specified
            - 'multiplier' (int): Molecule multiplier (e.g., 2 in "[2M+H]+")
        """
        # Remove outer whitespace
        adduct = adduct_str.strip()

        # Initialize result
        charge_explicit = False
        explicit_charge = 0
        composition = []
        mass_delta = 0.0
        multiplier = 1

        # Pattern 1: Explicit charge notation [...]z+/- or [...]+/-z
        # Examples: [M+H]+, [M+2H]2+, [2M-H]-, [M+Na]1+
        bracket_match = re.match(r"\[([^\]]+)\]\s*(\d*)([+-])", adduct)
        if bracket_match:
            inner, charge_num, charge_sign = bracket_match.groups()
            charge_explicit = True
            charge_magnitude = int(charge_num) if charge_num else 1
            explicit_charge = (
                charge_magnitude if charge_sign == "+" else -charge_magnitude
            )

            # Parse inner content: look for nM and adduct components
            # Pattern: optional number before M, then components
            inner = inner.strip()

            # Check for molecule multiplier (e.g., "2M+H")
            mult_match = re.match(r"(\d+)\s*M", inner)
            if mult_match:
                multiplier = int(mult_match.group(1))
                inner = re.sub(r"^\d+\s*M", "M", inner)  # Remove multiplier from string

            # Remove leading 'M' if present
            inner = re.sub(r"^M\s*", "", inner)

            # Parse remaining components (e.g., "+H", "+2Na-H", "+HCOOH-H")
            composition = self._parse_composition(inner)

        else:
            # Pattern 2: Composition-based notation (...)
            # Examples: (+H), (-2H), (+Na), (+HCOOH-H)
            # Remove parentheses
            inner = adduct.strip("() ")
            composition = self._parse_composition(inner)

            # Fallback: infer charge from composition
            # This is a heuristic and may not be accurate for all formats
            charge_explicit = False
            explicit_charge = self._infer_charge_from_composition(composition)

        # Calculate mass delta from composition using formula-based approach
        for sign_str, count, component in composition:
            sign = 1 if sign_str == "+" else -1
            component_mass = self._get_component_mass(component)
            mass_delta += sign * count * component_mass

        return {
            "charge": explicit_charge,
            "composition": composition,
            "mass_delta": mass_delta,
            "charge_explicit": charge_explicit,
            "multiplier": multiplier,
        }

    def _parse_composition(self, comp_str: str) -> List[tuple]:
        """
        Parse composition string into list of (sign, count, component) tuples.

        Examples:
        - "+H" -> [('+', 1, 'H')]
        - "+2Na-H" -> [('+', 2, 'Na'), ('-', 1, 'H')]
        - "+HCOOH-H" -> [('+', 1, 'HCOOH'), ('-', 1, 'H')]

        Parameters
        ----------
        comp_str : str
            Composition string (without brackets or parentheses)

        Returns
        -------
        List[tuple]
            List of (sign, count, component) tuples where:
            - sign: '+' or '-'
            - count: integer count
            - component: string identifier (e.g., 'H', 'Na', 'HCOOH')
        """
        if not comp_str:
            return []

        composition = []

        # Pattern: [+-]?(\d*)([A-Za-z0-9]+)
        # Matches: optional sign, optional count, component name
        pattern = r"([+-]?)(\d*)([A-Za-z0-9]+)"
        matches = re.findall(pattern, comp_str)

        for i, (sign_str, count_str, component) in enumerate(matches):
            if not component:
                continue

            # Determine sign: if first component has no sign, infer from leading character
            if i == 0 and not sign_str:
                sign_str = "+" if comp_str[0] != "-" else "-"
            elif not sign_str:
                sign_str = "+"  # Default to addition

            count = int(count_str) if count_str else 1
            composition.append((sign_str, count, component))

        return composition

    def _infer_charge_from_composition(self, composition: List[tuple]) -> int:
        """
        Infer charge state from composition as a heuristic fallback.

        This uses ion chemistry rules to estimate charge by looking at what species
        are ADDED to the molecule. Losses (sign='-') only affect charge when they
        represent deprotonation or loss of charged species.

        Charge contribution rules:
        - Adding cations (H+, Na+, K+, Li+, NH4+): +1 per ion added
        - Adding divalent cations (Mg2+, Ca2+, Fe2+): +2 per ion added
        - Adding anions (Cl-, Br-, I-, F-, HCOO-, CH3COO-): -1 per ion added
        - Adding neutral molecules (HCOOH, CH3COOH, H2O): 0 contribution
        - Losing H+ (deprotonation): -1 per H removed
        - Losing or adding other species: typically 0 charge effect

        Special cases handled:
        - (+HCOOH-H) -> HCOO- -> charge = -1 (neutral acid loses proton)
        - (+CH3COOH-H) -> CH3COO- -> charge = -1 (neutral acid loses proton)
        - (+Na-H) -> Na+ replaces H+ -> charge = 0 (cation exchange)
        - (-CH3COO) -> charge = 0 (losing acetate doesn't infer charge state)

        Parameters
        ----------
        composition : List[tuple]
            List of (sign, count, component) tuples

        Returns
        -------
        int
            Inferred charge state
        """
        total_charge = 0

        for sign_str, count, component in composition:
            sign = 1 if sign_str == "+" else -1

            # Determine charge contribution based on ion chemistry
            # IMPORTANT: Check longer strings before substrings (HCOOH before HCOO)

            # Neutral molecules (no charge contribution)
            if component in ["HCOOH", "CH3COOH", "H2O"]:
                total_charge += 0

            # Protons: always contribute to charge (gain or loss)
            elif component == "H":
                # Adding H+ increases charge by +1, losing H+ decreases by -1
                total_charge += sign * count * 1

            # Monovalent cations: only contribute when ADDED
            elif component in ["Na", "K", "Li", "NH4"]:
                if sign_str == "+":
                    total_charge += count * 1
                # Losing a cation doesn't infer charge state (could be neutral or charged starting point)

            # Divalent cations: only contribute when ADDED
            elif component in ["Mg", "Ca", "Fe", "Zn", "Cu"]:
                if sign_str == "+":
                    total_charge += count * 2

            # Anions: only contribute when ADDED
            elif component in ["Cl", "Br", "I", "F"]:
                if sign_str == "+":
                    total_charge += count * (-1)
            elif component in ["HCOO", "formate"]:
                if sign_str == "+":
                    total_charge += count * (-1)
            elif component in ["CH3COO", "acetate"]:
                if sign_str == "+":
                    total_charge += count * (-1)

            # Unknown components: no charge inference
            # (We don't know if it's ionic or neutral, so assume 0)

        return total_charge

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

        # Mass delta (normalized by common range 0-50 Da)
        mass_delta = np.array([parsed["mass_delta"] / 50.0], dtype=np.float32)

        # Concatenate features
        features = np.concatenate(
            [
                charge_mag,  # 1
                charge_sign,  # 2
                mass_delta,  # 1
            ]
        )

        # Keep continuous features as float32 for the adduct_encoder MLP
        return features.astype(np.float32)

    def extract_descriptor(
        self, adduct_str: str, neutral_exact_mass: float
    ) -> np.ndarray:
        """
        Extract physicochemical descriptor vector for adduct generalization.

        This descriptor-based approach enables generalization to unseen adduct strings
        by encoding the physical properties that affect ion mobility.

        Features extracted:
        - z_signed: signed charge state (e.g., +1, +2, -1)
        - z_abs: absolute charge magnitude
        - delta_mass: mass shift from adduct (Da)
        - mz: mass-to-charge ratio
        - polarity: charge polarity (+1 or -1)
        - charge_density: charge per mass unit proxy
        - ion_type_encoding: multi-hot encoding of ion composition (9 dims)

        Parameters
        ----------
        adduct_str : str
            Adduct string (e.g., "(+H)", "(-2H)", "(+Na)")
        neutral_exact_mass : float
            Exact neutral mass of the molecule in Daltons

        Returns
        -------
        np.ndarray
            Descriptor vector of shape (15,) with continuous physicochemical features
        """
        parsed = self.parse_adduct(adduct_str)
        z = int(parsed["charge"])
        z_abs = max(1, abs(z))  # Avoid divide-by-zero if parsing fails
        delta_mass = float(parsed["mass_delta"])
        ion_mass = neutral_exact_mass + delta_mass
        mz = ion_mass / z_abs

        # Core physicochemical features
        z_signed = float(z)
        polarity = 1.0 if z > 0 else -1.0
        charge_density = z_abs / max(1e-6, ion_mass)  # Charge per mass proxy

        # Base continuous features (6 dimensions)
        base = np.array(
            [z_signed, float(z_abs), delta_mass, mz, polarity, charge_density],
            dtype=np.float32,
        )

        # Ion type composition (multi-hot encoding, 9 dimensions)
        ion_enc = self.get_ion_type_encoding(parsed["composition"]).astype(np.float32)

        # Concatenate: 6 + 9 = 15 dimensions total
        descriptor = np.concatenate([base, ion_enc], axis=0)

        return descriptor
