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
  2  delta_mass (Da)                 (computed from tokens)
  3  mz (Da/e)                       (requires SMILES; respects multiplier)
  4  polarity (+1 or -1)
  5  charge_density_proxy            z_abs / ion_mass (requires SMILES; respects multiplier)

Neutral molecule polarizability proxy (1):
  6  alpha_molecule_A3               (requires SMILES)

Adduct token/radius moment proxies - SPLIT into additions and losses (18):
  Mass contributions:
    7   delta_mass_add               mass added from '+' tokens
    8   delta_mass_loss              mass removed from '-' tokens (positive magnitude)

  Atom count contributions:
    9   n_atoms_add                  total atoms added
    10  n_atoms_loss                 total atoms removed (positive magnitude)
    11  n_heavy_add                  heavy (non-H) atoms added
    12  n_heavy_loss                 heavy atoms removed (positive magnitude)

  Atomic number contributions:
    13  Z_sum_add                    sum of Z added
    14  Z_sum_loss                   sum of Z removed (positive magnitude)

  Van der Waals radius moments (split):
    15  rvdw1_add                    Σ n_e * r_vdw(e) for additions
    16  rvdw1_loss                   Σ n_e * r_vdw(e) for losses (positive magnitude)
    17  rvdw2_add                    Σ n_e * r_vdw(e)^2 for additions
    18  rvdw2_loss                   Σ n_e * r_vdw(e)^2 for losses (positive magnitude)
    19  rvdw3_add                    Σ n_e * r_vdw(e)^3 for additions
    20  rvdw3_loss                   Σ n_e * r_vdw(e)^3 for losses (positive magnitude)

  Net / targeted (kept at 4 dims to preserve vector length):
    21  delta_total_atoms            net atom change (add - loss)
    22  delta_heavy_atoms            net heavy atom change
    23  H_add                        number of H atoms added via '+' tokens (includes H+)
    24  H_loss                       number of H atoms removed via '-' tokens (e.g., -2H, -H2O contributes 2)

Loss fraction features (3) - requires SMILES:
  25  loss_mass_fraction             delta_mass_loss / (multiplier * neutral_mass)
  26  loss_heavy_fraction            n_heavy_loss / (multiplier * n_heavy_neutral)
  27  loss_H_fraction                H_loss / (multiplier * n_H_neutral)

Important behavior
------------------
- extract_descriptor(adduct_str, smiles) always returns the full 28D vector.
- extract_features(adduct_str, smiles=None) returns:
    * full 28D vector if smiles is provided
    * reduced vector (22D) if smiles is omitted (drops SMILES-dependent terms: mz, charge_density, alpha, loss fractions)
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
    element_delta: Dict[str, int]  # signed element count changes (net)
    delta_mass: float  # Da (token-based)


class AdductFeatureExtractor:
    """
    Vocab-free physicochemical descriptor extractor.

    Notes
    -----
    - Token parsing is formula-based (periodic-table only). No adduct identity tables.
    - Masses for non-proton elements use RDKit periodic table atomic weights; this is a proxy.
    """

    # Avogadro constant (exact SI definition)
    N_A = 6.02214076e23  # 1/mol
    CM3_TO_A3 = 1e24  # 1 cm^3 = 1e24 Å^3

    # Proton mass used for MS adduct bookkeeping (H+), in Daltons
    PROTON_MASS = 1.007276466812

    # Regex: bracket body + optional charge magnitude + sign
    _BRACKET_RE = re.compile(r"^\[(?P<body>[^\]]+)\](?P<zmag>\d*)(?P<sign>[+-])$")

    # Optional multiplier in body: 2M, 3M, ...
    _MULT_RE = re.compile(r"^(?P<mult>\d+)\s*M")

    # Token stream parser after leading "M": +Na-2H+H2O etc.
    _TOKEN_RE = re.compile(r"(?P<pm>[+-])(?P<count>\d*)(?P<tok>[A-Za-z0-9]+)")

    # Formula parser: element symbol + optional count (e.g., CH3COO -> C2 H3 O2)
    _FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

    def __init__(self, warn_on_unparsed_tokens: bool = True) -> None:
        self._ptable = Chem.GetPeriodicTable()
        self._warn_on_unparsed_tokens = bool(warn_on_unparsed_tokens)
        self._warned_tokens: set[str] = set()

    # -------------------------
    # RDKit molecule helpers
    # -------------------------
    @staticmethod
    def _mol_from_smiles(smiles: str) -> Chem.Mol:
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
        """Count total hydrogens (implicit+explicit) in neutral molecule from SMILES via formula parsing."""
        mol = self._mol_from_smiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        h_match = re.search(r"H(\d*)", formula)
        if not h_match:
            return 0
        h_count_str = h_match.group(1)
        return int(h_count_str) if h_count_str else 1

    # -------------------------
    # Formula parsing & masses
    # -------------------------
    def _parse_formula(self, token: str) -> Dict[str, int]:
        """Parse a token as a chemical formula into element counts."""
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
        Element mass provider using RDKit periodic table atomic weights (average masses).
        This is a proxy suitable for conditioning features.
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

        Special case: token == 'H' is treated as a proton (H+).
        """
        if token == "H":
            return float(self.PROTON_MASS)

        if not token_comp:
            # Could be 'e' or malformed; treat as 0 but optionally warn
            self._maybe_warn_unparsed(token)
            return 0.0

        m = 0.0
        for el, n in token_comp.items():
            em = self._element_mass(el)
            if em is None:
                warnings.warn(
                    f"Unknown element {el!r} in token {token!r}; treating that element mass as 0.",
                    UserWarning,
                )
                continue
            m += em * float(n)
        return float(m)

    def _maybe_warn_unparsed(self, token: str) -> None:
        if not self._warn_on_unparsed_tokens:
            return
        if token in self._warned_tokens:
            return
        self._warned_tokens.add(token)
        warnings.warn(
            f"Token {token!r} could not be parsed as a chemical formula; it will be ignored in proxy features.",
            UserWarning,
        )

    # -------------------------
    # Adduct parsing
    # -------------------------
    def parse_adduct(self, adduct_str: str) -> ParsedAdduct:
        """
        Parse a bracketed adduct string with explicit trailing charge.
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
            if not comp:
                self._maybe_warn_unparsed(tok)
                continue

            # Aggregate net element deltas (still useful occasionally)
            for el, n in comp.items():
                element_delta[el] = element_delta.get(el, 0) + sgn * cnt * n

            # Signed mass delta from token (token-based, consistent with split mass features)
            delta_mass += sgn * float(cnt) * self._token_mass(tok, comp)

        return ParsedAdduct(
            z_signed=int(z_signed),
            multiplier=int(multiplier),
            composition=composition,
            element_delta=element_delta,
            delta_mass=float(delta_mass),
        )

    # -------------------------
    # Adduct token proxies (vocab-free)
    # -------------------------
    def _adduct_token_proxies(
        self, composition: List[Tuple[str, int, str]]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute split add/loss proxies from *token-level* composition (avoids cancellation).

        Returns:
            proxies: 18D float32
            H_loss:  float (count of H atoms lost, positive magnitude)
            H_add:   float (count of H atoms added, positive magnitude)
        """
        mass_add = 0.0
        mass_loss = 0.0

        atoms_add = 0.0
        atoms_loss = 0.0
        heavy_add = 0.0
        heavy_loss = 0.0

        H_loss = 0.0
        H_add = 0.0

        Z_add = 0.0
        Z_loss = 0.0

        r1_add = 0.0
        r1_loss = 0.0
        r2_add = 0.0
        r2_loss = 0.0
        r3_add = 0.0
        r3_loss = 0.0

        for pm, cnt, tok in composition:
            is_add = pm == "+"
            cnt_i = int(cnt)

            tok_comp = self._parse_formula(tok)
            if not tok_comp:
                self._maybe_warn_unparsed(tok)
                continue

            tok_mass = self._token_mass(tok, tok_comp)
            if is_add:
                mass_add += cnt_i * tok_mass
            else:
                mass_loss += cnt_i * tok_mass

            for el, n_el in tok_comp.items():
                z = self._ptable.GetAtomicNumber(el)
                if z <= 0:
                    continue

                n_total = float(cnt_i * n_el)

                if is_add:
                    atoms_add += n_total
                    if el != "H":
                        heavy_add += n_total
                    else:
                        H_add += n_total
                else:
                    atoms_loss += n_total
                    if el != "H":
                        heavy_loss += n_total
                    else:
                        H_loss += n_total

                z_contrib = n_total * float(z)
                if is_add:
                    Z_add += z_contrib
                else:
                    Z_loss += z_contrib

                r = float(self._ptable.GetRvdw(z))
                r1 = n_total * r
                r2 = n_total * (r * r)
                r3 = n_total * (r * r * r)

                if is_add:
                    r1_add += r1
                    r2_add += r2
                    r3_add += r3
                else:
                    r1_loss += r1
                    r2_loss += r2
                    r3_loss += r3

        # Net values (kept as 2 dims) + targeted H add/loss (2 dims) to preserve length
        delta_total_atoms = atoms_add - atoms_loss
        delta_heavy_atoms = heavy_add - heavy_loss

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
                # Z sums (2)
                Z_add,
                Z_loss,
                # rvdw moments (6)
                r1_add,
                r1_loss,
                r2_add,
                r2_loss,
                r3_add,
                r3_loss,
                # Net/targeted (4) -> keep vector length stable
                delta_total_atoms,
                delta_heavy_atoms,
                H_add,
                H_loss,
            ],
            dtype=np.float32,
        )

        return proxies, float(H_loss), float(H_add)

    # -------------------------
    # Public feature APIs
    # -------------------------
    def extract_descriptor(self, adduct_str: str, smiles: str) -> np.ndarray:
        """
        Compute the full 28D descriptor vector (float32). Requires SMILES.
        """
        parsed = self.parse_adduct(adduct_str)

        z = int(parsed.z_signed)
        z_abs = max(1, abs(z))
        polarity = 1.0 if z > 0 else -1.0

        # Neutral / ion masses must respect the multiplier (e.g., [2M-H]2-)
        neutral_mass = self.neutral_exact_mass_from_smiles(smiles)
        neutral_mass_total = float(parsed.multiplier) * float(neutral_mass)
        ion_mass = float(neutral_mass_total + float(parsed.delta_mass))

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

        adduct_proxies, H_loss, _H_add = self._adduct_token_proxies(parsed.composition)

        # loss fractions (normalized by molecule size, respecting multiplier)
        delta_mass_loss = float(adduct_proxies[1])  # mass_loss
        n_heavy_loss = float(adduct_proxies[5])  # heavy_loss

        loss_mass_fraction = delta_mass_loss / max(1e-6, neutral_mass_total)

        n_heavy_neutral = float(self.neutral_heavy_atom_count(smiles))
        n_heavy_neutral_total = float(parsed.multiplier) * max(1.0, n_heavy_neutral)
        loss_heavy_fraction = n_heavy_loss / n_heavy_neutral_total

        n_H_neutral = float(self.neutral_hydrogen_count(smiles))
        n_H_neutral_total = float(parsed.multiplier) * max(1.0, n_H_neutral)
        loss_H_fraction = float(H_loss) / n_H_neutral_total

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
          returns a reduced vector (22D) that excludes SMILES-dependent features:
            Core-light (4): z_signed, z_abs, delta_mass, polarity
            Adduct proxies (18): split mass/counts/Z/radius moments + net/heavy + H_add/H_loss
        """
        parsed = self.parse_adduct(adduct_str)

        z = int(parsed.z_signed)
        z_abs = float(max(1, abs(z)))
        polarity = 1.0 if z > 0 else -1.0
        delta_mass = float(parsed.delta_mass)

        adduct_proxies, _H_loss, _H_add = self._adduct_token_proxies(parsed.composition)

        if smiles is not None:
            return self.extract_descriptor(adduct_str, smiles)

        reduced = np.concatenate(
            [
                np.array(
                    [float(z), z_abs, delta_mass, polarity], dtype=np.float32
                ),  # 4
                adduct_proxies.astype(np.float32),  # 18 -> 22
            ],
            axis=0,
        ).astype(np.float32)

        return reduced

    # -------------------------
    # Optional: reference-adduct utilities (vocab-free)
    # -------------------------
    @staticmethod
    def reference_adduct_for_charge(z_signed: int) -> str:
        """Return a simple reference adduct string based only on charge sign."""
        return "[M+H]+" if z_signed > 0 else "[M-H]-"

    def extract_descriptor_with_reference(
        self, adduct_str: str, smiles: str, include_delta: bool = True
    ) -> np.ndarray:
        """
        Optional helper: concatenate descriptor(adduct), descriptor(reference), and optionally their difference.

        This is often useful for learning adduct-induced shifts, especially for neutral-loss and exchange adducts.
        Output shape:
          - if include_delta: 28 * 3 = 84
          - else:            28 * 2 = 56
        """
        parsed = self.parse_adduct(adduct_str)
        ref = self.reference_adduct_for_charge(parsed.z_signed)

        f = self.extract_descriptor(adduct_str, smiles)
        f_ref = self.extract_descriptor(ref, smiles)

        if not include_delta:
            return np.concatenate([f, f_ref], axis=0).astype(np.float32)

        f_delta = (f - f_ref).astype(np.float32)
        return np.concatenate([f, f_ref, f_delta], axis=0).astype(np.float32)
