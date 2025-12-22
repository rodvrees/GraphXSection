"""
Adduct feature encoding for physicochemical descriptor representation (vocab-free).

Design goals
------------
- No adduct vocabularies (no token→SMILES maps, no ion-type multi-hot, no carrier tables).
- All features computable from:
  (1) the adduct string itself (parsed as signed chemical-formula tokens), and
  (2) the neutral molecule SMILES (RDKit).
- Descriptor is continuous and LOAO-friendly (no identity leakage).

Supported adduct formats
------------------------
Bracketed adducts with explicit trailing charge, e.g.:
  [M+H]+
  [M+Na-2H]-
  [M-H]-
  [M-CH3]-
  [M+H-H2O]+
  [M+HCOO]-
  [M+NH4]+
  [M+CH3COO]-
  [2M-H]2-

Descriptor output (fixed length: 28)
------------------------------------
Core (6):
  0  z_signed
  1  z_abs
  2  delta_mass (Da)
  3  mz (Da/e)               (requires SMILES)
  4  polarity (+1 or -1)
  5  charge_density_proxy    z_abs / ion_mass (requires SMILES)

Neutral molecule polarizability proxy (1):
  6  alpha_molecule_A3       (requires SMILES)

Adduct element/radius moment proxies - SPLIT into additions and losses (18):
  Mass contributions:
    7   delta_mass_add      mass added from '+' tokens
    8   delta_mass_loss     mass removed from '-' tokens (positive magnitude)

  Atom count contributions:
    9   n_atoms_add         total atoms added
    10  n_atoms_loss        total atoms removed (positive magnitude)
    11  n_heavy_add         heavy (non-H) atoms added
    12  n_heavy_loss        heavy atoms removed (positive magnitude)

  Atomic number contributions:
    13  Z_sum_add           sum of Z added
    14  Z_sum_loss          sum of Z removed (positive magnitude)

  Van der Waals radius moments (split):
    15  rvdw1_add           Σ n_e * r_vdw(e) for additions
    16  rvdw1_loss          |Σ n_e * r_vdw(e)| for losses
    17  rvdw2_add           Σ n_e * r_vdw(e)^2 for additions
    18  rvdw2_loss          |Σ n_e * r_vdw(e)^2| for losses
    19  rvdw3_add           Σ n_e * r_vdw(e)^3 for additions
    20  rvdw3_loss          |Σ n_e * r_vdw(e)^3| for losses

  Net values (for backward compatibility / convenience):
    21  delta_total_atoms   net atom change (add - loss)
    22  delta_heavy_atoms   net heavy atom change
    23  delta_Z_sum         net Z change
    24  delta_rvdw_1        net r_vdw moment

Loss fraction features (3) - requires SMILES:
  25  loss_mass_fraction    delta_mass_loss / neutral_mass
  26  loss_heavy_fraction   n_heavy_loss / n_heavy_neutral
  27  loss_H_fraction       n_H_loss / n_H_neutral

Notes on physics proxies
------------------------
- Neutral "polarizability proxy" is derived from RDKit Wildman–Crippen molar refractivity (MR),
  converted to a polarizability volume proxy (Å^3) using:
      alpha = (3 * MR) / (4π N_A) * 1e24
  MR is cm^3/mol, N_A is Avogadro constant.

- Adduct “polarizability-like” effects are approximated without identifying ions by using
  element-wise van der Waals radius moments, computed from signed element deltas implied
  by the adduct string.

Important behavior
------------------
- extract_descriptor(adduct_str, smiles) always returns the full 28D vector.
- extract_features(adduct_str, smiles=None) returns:
    * full 28D vector if smiles is provided
    * reduced vector if smiles is omitted (drops the SMILES-dependent terms: mz, charge_density, alpha, loss fractions)
  This makes the function usable in both contexts without any adduct vocab.

Rationale for split features
-----------------------------
For composite adducts like [M+Na-2H]- or [M+H-H2O]+, the net delta can be small while
the underlying event involves both addition and removal. These processes can drive CCS
in different directions (e.g., loss tends to compact, addition tends to expand). By
providing separate features for additions and losses, the model can learn these effects
independently without cancellation in the net value.

Rationale for loss fraction features
-------------------------------------
Large neutral losses relative to molecule size tend to strongly shift CCS (often negative
due to compaction). The loss fraction features let the network learn this relationship
without needing to identify specific fragments:
- loss_mass_fraction: How much mass is lost relative to total neutral mass
- loss_heavy_fraction: How many heavy atoms are lost relative to total heavy atoms
- loss_H_fraction: How many hydrogens are lost relative to total hydrogens
These are normalized by molecule size, making them comparable across different molecules.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@dataclass(frozen=True)
class ParsedAdduct:
    """Structured parse output for an adduct string."""

    z_signed: int
    multiplier: int
    composition: List[Tuple[str, int, str]]  # (sign, count, token)
    element_delta: Dict[str, int]  # signed element count changes
    delta_mass: float  # Da


class AdductFeatureExtractor:
    """
    Vocab-free physicochemical descriptor extractor.
    """

    # Avogadro constant (exact SI definition)
    N_A = 6.02214076e23  # 1/mol
    CM3_TO_A3 = 1e24  # 1 cm^3 = 1e24 Å^3

    # Proton mass used for MS adduct bookkeeping (H+), in Daltons
    PROTON_MASS = 1.007276466812

    # Regex: bracket body + optional charge magnitude + sign, anchored
    _BRACKET_RE = re.compile(r"^\[(?P<body>[^\]]+)\](?P<zmag>\d*)(?P<sign>[+-])$")

    # Optional multiplier in body: 2M, 3M, ...
    _MULT_RE = re.compile(r"^(?P<mult>\d+)\s*M")

    # Token stream parser after leading "M": +Na-2H+H2O etc.
    _TOKEN_RE = re.compile(r"(?P<pm>[+-])(?P<count>\d*)(?P<tok>[A-Za-z0-9]+)")

    # Formula parser: element symbol + optional count (e.g., CH3COO -> C2 H3 O2)
    _FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

    def __init__(self) -> None:
        self._ptable = Chem.GetPeriodicTable()

    # -------------------------
    # RDKit molecule helpers
    # -------------------------
    @staticmethod
    def _mol_from_smiles(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        return mol

    def neutral_exact_mass_from_smiles(self, smiles: str) -> float:
        """Neutral exact mass (monoisotopic) from SMILES via RDKit."""
        mol = self._mol_from_smiles(smiles)
        return float(rdMolDescriptors.CalcExactMolWt(mol))

    def molecule_alpha_A3_from_smiles(self, smiles: str) -> float:
        """
        Neutral polarizability proxy (Å^3) from RDKit Wildman–Crippen MR.

        RDKit returns MR in cm^3/mol.
        alpha (cm^3/molecule) = 3*MR / (4π N_A)
        Convert cm^3 -> Å^3 using 1 cm^3 = 1e24 Å^3.
        """
        mol = self._mol_from_smiles(smiles)
        _, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)  # (logP, MR)
        mr = float(mr)
        alpha_cm3_per_molecule = (3.0 * mr) / (4.0 * np.pi * self.N_A)
        return float(alpha_cm3_per_molecule * self.CM3_TO_A3)

    def neutral_heavy_atom_count(self, smiles: str) -> int:
        """Count heavy (non-hydrogen) atoms in neutral molecule from SMILES."""
        mol = self._mol_from_smiles(smiles)
        return int(mol.GetNumHeavyAtoms())

    def neutral_hydrogen_count(self, smiles: str) -> int:
        """Count total hydrogens (explicit + implicit) in neutral molecule from SMILES."""
        mol = self._mol_from_smiles(smiles)
        # Get molecular formula and extract H count
        formula = rdMolDescriptors.CalcMolFormula(mol)
        # Parse H count from formula (e.g., "C2H6O" -> 6)
        import re

        h_match = re.search(r"H(\d*)", formula)
        if h_match:
            h_count_str = h_match.group(1)
            return int(h_count_str) if h_count_str else 1
        return 0  # No hydrogens in molecule

    # -------------------------
    # Formula parsing & masses
    # -------------------------
    def _parse_formula(self, token: str) -> Dict[str, int]:
        """
        Parse a token as a chemical formula into element counts.

        Examples:
          H        -> {H:1}
          H2O      -> {H:2, O:1}
          CH3COO   -> {C:2, H:3, O:2}
          HCOO     -> {H:1, C:1, O:2}
          Na       -> {Na:1}
          NH4      -> {N:1, H:4}

        If parsing yields no valid elements, returns {}.
        """
        matches = self._FORMULA_RE.findall(token)
        if not matches:
            return {}

        comp: Dict[str, int] = {}
        for el, n_str in matches:
            if not el:
                continue
            n = int(n_str) if n_str else 1
            comp[el] = comp.get(el, 0) + n
        return comp

    def _element_mass(self, element: str) -> Optional[float]:
        """
        Vocab-free element mass provider.

        Uses RDKit periodic table atomic weights (average masses).
        This is sufficient for descriptor conditioning; exact monoisotopic masses are not required.
        """
        try:
            z = self._ptable.GetAtomicNumber(element)
            if z <= 0:
                return None
            return float(self._ptable.GetAtomicWeight(z))
        except Exception:
            return None

    def _token_mass(self, token: str, token_comp: Dict[str, int]) -> float:
        """
        Compute mass contribution for a token treated as a formula.

        Important: If token is exactly 'H' as an adduct component, treat as proton mass (H+).
        Otherwise sum element masses from the periodic table.
        """
        if token == "H":
            return float(self.PROTON_MASS)

        if not token_comp:
            # Could be 'e' or malformed token; ignore with warning
            warnings.warn(
                f"Token {token!r} could not be parsed as a formula; mass set to 0.0",
                UserWarning,
            )
            return 0.0

        m = 0.0
        for el, n in token_comp.items():
            em = self._element_mass(el)
            if em is None:
                warnings.warn(
                    f"Unknown element {el!r} in token {token!r}; mass set to 0.0 for that element",
                    UserWarning,
                )
                continue
            m += em * float(n)
        return float(m)

    # -------------------------
    # Adduct parsing
    # -------------------------
    def parse_adduct(self, adduct_str: str) -> ParsedAdduct:
        """
        Parse a bracketed adduct string with explicit trailing charge.

        Returns:
        - z_signed from trailing charge
        - multiplier (n in nM)
        - composition list of (sign, count, token)
        - element_delta: signed element count changes aggregated over tokens
        - delta_mass: signed mass delta computed from formula tokens
        """
        s = adduct_str.strip()
        m = self._BRACKET_RE.match(s)
        if not m:
            raise ValueError(
                f"Unsupported adduct format (expected '[...]+' or '[...]-'): {adduct_str!r}"
            )

        body = m.group("body").strip()
        zmag = int(m.group("zmag")) if m.group("zmag") else 1
        zsign = +1 if m.group("sign") == "+" else -1
        z_signed = zmag * zsign

        # multiplier: nM
        multiplier = 1
        mm = self._MULT_RE.match(body)
        if mm:
            multiplier = int(mm.group("mult"))
            body = self._MULT_RE.sub("M", body, count=1).strip()

        if not body.startswith("M"):
            raise ValueError(f"Adduct body must start with 'M': {adduct_str!r}")

        tail = body[1:]  # after leading M

        composition: List[Tuple[str, int, str]] = []
        element_delta: Dict[str, int] = {}
        delta_mass = 0.0

        for mt in self._TOKEN_RE.finditer(tail):
            pm = mt.group("pm")
            cnt = int(mt.group("count")) if mt.group("count") else 1
            tok = mt.group("tok")

            composition.append((pm, cnt, tok))

            sgn = +1 if pm == "+" else -1
            comp = self._parse_formula(tok)

            # Aggregate signed element deltas
            for el, n in comp.items():
                element_delta[el] = element_delta.get(el, 0) + sgn * cnt * n

            # Mass delta from the token treated as formula
            delta_mass += sgn * float(cnt) * self._token_mass(tok, comp)

        return ParsedAdduct(
            z_signed=int(z_signed),
            multiplier=int(multiplier),
            composition=composition,
            element_delta=element_delta,
            delta_mass=float(delta_mass),
        )

    # -------------------------
    # Adduct summary proxies (vocab-free)
    # -------------------------
    def _adduct_radius_moments(
        self, composition: List[Tuple[str, int, str]]
    ) -> Tuple[np.ndarray, float]:
        """
        Compute adduct proxies from composition tokens using periodic table radii.

        Now processes tokens directly to avoid cancellation:
        - For [M+H-H2O]+: sees "+H" and "-H2O" separately, not just net "-H -O"
        - Returns SPLIT features (18) to distinguish additions from removals

        Features:
        - Mass: delta_mass_add, delta_mass_loss
        - Counts: n_atoms_add, n_atoms_loss, n_heavy_add, n_heavy_loss
        - Atomic numbers: Z_sum_add, Z_sum_loss
        - Radius moments: rvdw1_add, rvdw1_loss, rvdw2_add, rvdw2_loss, rvdw3_add, rvdw3_loss
        - Net values: delta_total_atoms, delta_heavy_atoms, delta_Z_sum, delta_rvdw_1

        All 'loss' values are positive magnitudes (absolute values).

        Args:
            composition: List of (pm, cnt, tok) where pm is '+' or '-'

        Returns:
            (adduct_proxies, n_H_loss): 18D array and H loss count
        """
        # Initialize addition and loss accumulators
        mass_add = 0.0
        mass_loss = 0.0

        atoms_add = 0.0
        atoms_loss = 0.0
        heavy_add = 0.0
        heavy_loss = 0.0

        n_H_loss = 0.0  # Track hydrogen losses separately
        n_H_add = 0.0   # Track hydrogen additions too

        Z_add = 0.0
        Z_loss = 0.0

        r1_add = 0.0
        r1_loss = 0.0
        r2_add = 0.0
        r2_loss = 0.0
        r3_add = 0.0
        r3_loss = 0.0

        # Process each token in the composition
        for pm, cnt, tok in composition:
            is_addition = (pm == '+')
            
            # Parse the token as a formula
            tok_comp = self._parse_formula(tok)
            if not tok_comp:
                continue
            
            # Get token mass
            tok_mass = self._token_mass(tok, tok_comp)
            
            # Accumulate mass
            if is_addition:
                mass_add += cnt * tok_mass
            else:
                mass_loss += cnt * tok_mass
            
            # Process each element in the token
            for el, n_el in tok_comp.items():
                z = self._ptable.GetAtomicNumber(el)
                if z <= 0:
                    continue
                
                # Total atoms of this element in this token instance
                n_total = float(cnt * n_el)
                
                # Atom count contributions
                if is_addition:
                    atoms_add += n_total
                    if el != "H":
                        heavy_add += n_total
                    else:
                        n_H_add += n_total
                else:
                    atoms_loss += n_total
                    if el != "H":
                        heavy_loss += n_total
                    else:
                        n_H_loss += n_total
                
                # Atomic number contributions
                z_contrib = n_total * float(z)
                if is_addition:
                    Z_add += z_contrib
                else:
                    Z_loss += z_contrib
                
                # van der Waals radius (Å) from periodic table
                r = float(self._ptable.GetRvdw(z))
                r1_contrib = n_total * r
                r2_contrib = n_total * (r * r)
                r3_contrib = n_total * (r * r * r)
                
                if is_addition:
                    r1_add += r1_contrib
                    r2_add += r2_contrib
                    r3_add += r3_contrib
                else:
                    r1_loss += r1_contrib
                    r2_loss += r2_contrib
                    r3_loss += r3_contrib

        # Compute net values (for backward compatibility and convenience)
        delta_total_atoms = atoms_add - atoms_loss
        delta_heavy_atoms = heavy_add - heavy_loss
        delta_Z_sum = Z_add - Z_loss
        delta_rvdw_1 = r1_add - r1_loss

        proxies = np.array(
            [
                # Mass (2)
                mass_add,
                mass_loss,
                # Atom counts (4)
                atoms_add,
                atoms_loss,
                heavy_add,
                heavy_loss,
                # Atomic numbers (2)
                Z_add,
                Z_loss,
                # Radius moments (6)
                r1_add,
                r1_loss,
                r2_add,
                r2_loss,
                r3_add,
                r3_loss,
                # Net values (4)
                delta_total_atoms,
                delta_heavy_atoms,
                delta_Z_sum,
                delta_rvdw_1,
            ],
            dtype=np.float32,
        )

        return proxies, float(n_H_loss)

    # -------------------------
    # Public feature APIs
    # -------------------------
    def extract_descriptor(self, adduct_str: str, smiles: str) -> np.ndarray:
        """
        Compute the full 28D descriptor vector (float32). Requires SMILES.

        Output:
          Core (6): z_signed, z_abs, delta_mass, mz, polarity, charge_density
          Molecule (1): alpha_molecule_A3
          Adduct split (18): mass, counts, Z, radius moments (all split into add/loss + net)
          Loss fractions (3): loss_mass_fraction, loss_heavy_fraction, loss_H_fraction
        """
        parsed = self.parse_adduct(adduct_str)

        z = int(parsed.z_signed)
        z_abs = max(1, abs(z))
        polarity = 1.0 if z > 0 else -1.0

        neutral_mass = self.neutral_exact_mass_from_smiles(smiles)
        ion_mass = float(neutral_mass + parsed.delta_mass)
        mz = float(ion_mass / float(z_abs))
        charge_density = float(z_abs / max(1e-6, ion_mass))

        core = np.array(
            [
                float(z),
                float(z_abs),
                float(parsed.delta_mass),
                mz,
                polarity,
                charge_density,
            ],
            dtype=np.float32,
        )

        alpha_mol = float(self.molecule_alpha_A3_from_smiles(smiles))
        alpha = np.array([alpha_mol], dtype=np.float32)

        adduct_proxies, n_H_loss = self._adduct_radius_moments(
            parsed.composition
        )  # (18,), float

        # Extract loss values from adduct_proxies for fraction calculation
        delta_mass_loss = float(adduct_proxies[1])  # index 1
        n_heavy_loss = float(adduct_proxies[5])  # index 5

        # Compute loss fractions (normalized by molecule size)
        # Use max(1e-6, ...) to avoid division by zero
        loss_mass_fraction = delta_mass_loss / max(1e-6, neutral_mass)

        n_heavy_neutral = float(self.neutral_heavy_atom_count(smiles))
        loss_heavy_fraction = n_heavy_loss / max(1.0, n_heavy_neutral)

        n_H_neutral = float(self.neutral_hydrogen_count(smiles))
        loss_H_fraction = n_H_loss / max(1.0, n_H_neutral)

        loss_fractions = np.array(
            [loss_mass_fraction, loss_heavy_fraction, loss_H_fraction],
            dtype=np.float32,
        )

        return np.concatenate(
            [core, alpha, adduct_proxies, loss_fractions], axis=0
        ).astype(np.float32)

    def extract_features(
        self, adduct_str: str, smiles: Optional[str] = None
    ) -> np.ndarray:
        """
        Feature vector for adduct conditioning, with optional SMILES.

        If `smiles` is provided:
          returns the full 28D descriptor (same as extract_descriptor)

        If `smiles` is None:
          returns a reduced vector that excludes SMILES-dependent features
          (mz, charge_density, alpha, loss_fractions):
            Core-light (4): z_signed, z_abs, delta_mass, polarity
            Adduct proxies (18): split mass/counts/Z/radius moments
          Total: 22D without SMILES-dependent features
        """
        parsed = self.parse_adduct(adduct_str)

        z = int(parsed.z_signed)
        z_abs = float(max(1, abs(z)))
        polarity = 1.0 if z > 0 else -1.0
        delta_mass = float(parsed.delta_mass)

        adduct_proxies, _ = self._adduct_radius_moments(
            parsed.composition
        )  # 18 dims, ignore H_loss

        if smiles is not None:
            return self.extract_descriptor(adduct_str, smiles)

        # Reduced, SMILES-free feature set (22 dims)
        # Omit: mz, charge_density, alpha_molecule, loss_fractions (all require SMILES)
        reduced = np.concatenate(
            [
                np.array(
                    [float(z), z_abs, delta_mass, polarity], dtype=np.float32
                ),  # 4
                adduct_proxies.astype(np.float32),  # 18 -> total 22
            ],
            axis=0,
        ).astype(np.float32)

        return reduced
