`atomtype_parameters_extended.in` is MOBCAL's standard atom-type Lennard-Jones
table (H/C/O/Na+/Si/S/Fe/P/F/Cl/N rows copied verbatim from `isicle`'s bundled
`resources/lennard_jones.txt`, which itself matches `mobcal-shm`'s own
upstream default `atomtype_parameters.in`) with two added rows: **Br and I**.

**These two rows are a derived approximation, not a literature value.**
Neither of MOBCAL's bundled tables (this one, or the alternate "-iu" legacy
table) covers anything past Cl -- confirmed directly, this fails at runtime
with `type not defined for atom number N` for any Br/I-containing molecule.
A literature search for a published Br/I parameter set compatible with this
specific potential form (MOBCAL's TM/MOBIL2 Lennard-Jones + ion-induced-dipole
model) came up empty.

Derivation:
1. Pulled UFF (Rappe et al. 1992) van der Waals parameters `x1` (distance)
   and `D1` (well depth, kcal/mol) directly from RDKit's own upstream source
   (`Code/ForceField/UFF/Params.cpp`, fetched from github.com/rdkit/rdkit),
   not retyped from any paper, for H, C, N, O, F, Cl, S, P, Si and for Br, I.
2. Confirmed MOBCAL's `atomtype_parameters.in` has no other hardcoded
   per-element table anywhere in `mobcal-shm`'s C source (`grep -rl polariz`
   over the whole tree: no hits) -- the EOLJ/ROLJ/RHS columns are the only
   place per-element physics lives, so extending this one file is sufficient.
3. Transferred UFF's Br/I values onto MOBCAL's own **Cl** row by ratio
   (same halogen group, the adjacent, already-fitted MOBCAL element):
     ROLJ_X = ROLJ_Cl * (x1_X / x1_Cl)
     EOLJ_X = EOLJ_Cl * (D1_X / D1_Cl)
   RHS scaled the same way. UFF values used (x1 Å, D1 kcal/mol):
   Cl (3.947, 0.227), Br (4.189, 0.251), I (4.500, 0.339).
4. Sanity check: this produces a monotonic Cl -> Br -> I increase in both
   EOLJ and ROLJ, consistent with the real polarizability trend
   (Cl ~2.2, Br ~3.05, I ~5.35 A^3) -- not a large jump, but the right
   direction and rough relative shape.
5. Spot-checked (not a real calibration -- only 2 points, opposite sign
   error, no MD/frame-averaging in this quick check) against two small
   Br-containing CCSBase drift-tube/N2 references:
     - brompheniramine-like (C[NH+](C)CCC(c1ccc(Br)cc1)c1ccccn1):
       predicted 179.9 A^2 vs experimental 165.1 A^2 (+9.0%)
     - dibromoaniline-like (C[NH+](Cc1cc(Br)cc(Br)c1N)C1CCCCC1):
       predicted 162.9 A^2 vs experimental 167.9 A^2 (-3.0%)
   No consistent bias direction from 2 points -- not enough to fit a
   trustworthy global scale correction, and doing so from 2 conflicting-sign
   points would be false precision. Both are within the range of scatter
   this coarse, single-static-geometry check would produce even for a
   well-established element, so this is evidence of "not grossly wrong",
   not "validated".

**Use accordingly**: fine for what `predict_candidate_ccs_md.py` actually
needs -- ranking a molecule's own candidates against each other by relative
CCS to redo the peak assignment (a systematic per-element bias applies
near-identically to every candidate of the same molecule and mostly cancels
in that comparison). Do not treat Br/I-containing absolute CCS predictions
from this table as validated for any other purpose without recalibrating
against more reference compounds.
