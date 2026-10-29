"""Structure standardisation.

A documented, versioned pipeline. Nothing here strips fragments blindly or
discards stereochemistry: both transformations can change the chemical entity
being represented, and stereochemistry is information a structure-based model
may need.

Two levels of identity are produced, and they are not the same thing:

  exact identity     InChIKey of the standardised structure, stereochemistry
                     included. Used to merge records describing the same
                     defined structure.
  scaffold identity  Bemis-Murcko scaffold of the standardised structure.
                     Used to group related compounds for leakage-resistant
                     splits. Two stereoisomers share a scaffold while remaining
                     chemically distinct.

Standardisation order, applied with RDKit:
  1. parse               reject malformed SMILES into a review table
  2. metal disconnection break covalent bonds to metals
  3. normalisation       functional-group normalisation
  4. fragment handling   choose the largest organic fragment (salts, solvates)
  5. charge normalisation neutralise where chemically sensible
  6. tautomer canonical  one representative tautomer, deterministically chosen
  7. identifiers         canonical SMILES, isomeric SMILES, InChIKey, formula,
                         molecular weight, scaffold

Every step is recorded per compound in `standardization_notes` so a surprising
result can be traced to the transformation that produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import inchi as rd_inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

LOG = logging.getLogger("standardize")

# RDKit's parse failures are captured and reported explicitly; silence the
# console spew so the pipeline's own report is the single source of truth.
RDLogger.DisableLog("rdApp.*")

STANDARDIZATION_VERSION = "std-v1.0.0"


def _stereo_preserving_params() -> rdMolStandardize.CleanupParameters:
    """Cleanup parameters that do not discard stereochemistry.

    RDKit's TautomerEnumerator defaults to tautomerRemoveSp3Stereo=True and
    tautomerRemoveBondStereo=True, which silently strips stereocentres during
    canonicalisation: L- and D-alanine would otherwise standardise to the same
    InChIKey and be merged into one compound. Stereochemistry is information a
    structure-based model may need, so it is kept.
    """
    params = rdMolStandardize.CleanupParameters()
    params.tautomerRemoveSp3Stereo = False
    params.tautomerRemoveBondStereo = False
    return params

# Status values written to `structure_status`.
STATUS_OK = "standardized"
STATUS_PARSE_FAILED = "parse_failed"
STATUS_EMPTY = "no_structure"
STATUS_MULTI_FRAGMENT = "standardized_from_mixture"
STATUS_NO_ORGANIC_FRAGMENT = "no_organic_fragment"
STATUS_STANDARDIZE_FAILED = "standardize_failed"


@dataclass
class StandardizationResult:
    """Outcome of standardising one input structure."""

    input_smiles: str | None
    status: str
    canonical_smiles: str | None = None
    isomeric_smiles: str | None = None
    inchi_key: str | None = None
    inchi: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    scaffold_smiles: str | None = None
    # First block of the InChIKey: the connectivity skeleton, shared by all
    # stereoisomers of a structure. Used for grouping, never as exact identity.
    connectivity_key: str | None = None
    n_stereocenters: int | None = None
    n_unassigned_stereocenters: int | None = None
    n_fragments: int | None = None
    standardization_version: str = STANDARDIZATION_VERSION
    notes: list[str] = field(default_factory=list)
    # Structured counterpart to `notes`. Each entry is
    # {"type": <code>, "detail": <str>} so the curation ledger can enumerate
    # transformations by type instead of parsing free text.
    transformations: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_OK, STATUS_MULTI_FRAGMENT)


def _is_organic(mol: Chem.Mol) -> bool:
    """A fragment containing at least one carbon bonded to something."""
    return any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def murcko_scaffold(mol: Chem.Mol, *, include_chirality: bool = False) -> str | None:
    """Bemis-Murcko scaffold SMILES.

    Chirality is excluded by default: the scaffold is a grouping key for split
    assignment, and two stereoisomers must group together so that they cannot
    be separated across a train/test boundary.

    An acyclic molecule has an empty Murcko scaffold. Rather than collapsing
    every acyclic compound into one giant group, such compounds are given no
    scaffold here and handled by the splitter as singleton groups.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("scaffold failed: %s", exc)
        return None


def standardize_smiles(smiles: str | None) -> StandardizationResult:
    """Run the documented standardisation pipeline on one SMILES string."""
    if smiles is None or not str(smiles).strip():
        return StandardizationResult(input_smiles=smiles, status=STATUS_EMPTY)

    raw = str(smiles).strip()
    notes: list[str] = []
    transformations: list[dict] = []

    mol = Chem.MolFromSmiles(raw, sanitize=True)
    if mol is None:
        return StandardizationResult(
            input_smiles=raw,
            status=STATUS_PARSE_FAILED,
            notes=["RDKit could not parse the SMILES"],
        )

    params = _stereo_preserving_params()

    try:
        # 2. metal disconnection
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)

        # 3. normalisation + general cleanup
        mol = rdMolStandardize.Cleanup(mol, params)

        # Counted AFTER metal disconnection, so that a covalently written salt
        # such as CC(=O)O[Na] is recognised as the mixture it is.
        n_fragments = len(Chem.GetMolFrags(mol, asMols=False))

        # 4. fragment handling - largest ORGANIC fragment, not blind stripping
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            organic = [f for f in frags if _is_organic(f)]
            if not organic:
                return StandardizationResult(
                    input_smiles=raw,
                    status=STATUS_NO_ORGANIC_FRAGMENT,
                    n_fragments=n_fragments,
                    notes=["mixture contained no organic fragment"],
                )
            # Largest by heavy-atom count; ties broken by canonical SMILES so
            # the choice is deterministic across runs.
            chosen = sorted(
                organic,
                key=lambda m: (-m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)),
            )[0]
            discarded = [
                Chem.MolToSmiles(f) for f in frags if f is not chosen
            ]
            notes.append(
                f"multi-fragment input; kept largest organic fragment, "
                f"discarded: {', '.join(sorted(discarded))}"
            )
            transformations.append(
                {
                    "type": "fragment_discarded",
                    "detail": "; ".join(sorted(discarded)),
                    "kept": Chem.MolToSmiles(chosen),
                }
            )
            mol = chosen

        # 5. charge normalisation
        mol = rdMolStandardize.Uncharger().uncharge(mol)

        # 6. canonical tautomer, deterministically chosen, stereo preserved
        enumerator = rdMolStandardize.TautomerEnumerator(params)
        mol = enumerator.Canonicalize(mol)

        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as exc:  # noqa: BLE001
        return StandardizationResult(
            input_smiles=raw,
            status=STATUS_STANDARDIZE_FAILED,
            n_fragments=locals().get("n_fragments"),
            notes=[f"standardisation raised: {exc}"],
        )

    # 7. identifiers. Stereochemistry is preserved in the isomeric form and
    # the InChIKey; the canonical form is kept without it for grouping.
    try:
        isomeric = Chem.MolToSmiles(mol, isomericSmiles=True)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        inchi_str = rd_inchi.MolToInchi(mol)
        inchi_key = rd_inchi.InchiToInchiKey(inchi_str) if inchi_str else None
        formula = rdMolDescriptors.CalcMolFormula(mol)
        weight = float(Descriptors.MolWt(mol))
        centres = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
        n_stereo = len(centres)
        n_unassigned = sum(1 for _, code in centres if code == "?")
        scaffold = murcko_scaffold(mol)
        connectivity = inchi_key.split("-")[0] if inchi_key else None
    except Exception as exc:  # noqa: BLE001
        return StandardizationResult(
            input_smiles=raw,
            status=STATUS_STANDARDIZE_FAILED,
            n_fragments=n_fragments,
            notes=[f"identifier generation raised: {exc}"],
        )

    if n_unassigned:
        transformations.append(
            {
                "type": "stereocentre_unassigned",
                "detail": f"{n_unassigned} of {n_stereo} stereocentre(s) unassigned in the source structure",
            }
        )
    if canonical != raw:
        transformations.append(
            {"type": "structure_standardized", "detail": f"{raw} -> {isomeric}"}
        )

    status = STATUS_MULTI_FRAGMENT if n_fragments > 1 else STATUS_OK
    return StandardizationResult(
        input_smiles=raw,
        status=status,
        canonical_smiles=canonical,
        isomeric_smiles=isomeric,
        inchi_key=inchi_key,
        inchi=inchi_str,
        molecular_formula=formula,
        molecular_weight=weight,
        scaffold_smiles=scaffold,
        connectivity_key=connectivity,
        n_stereocenters=n_stereo,
        n_unassigned_stereocenters=n_unassigned,
        n_fragments=n_fragments,
        notes=(
            notes + [f"{n_unassigned} stereocentre(s) unassigned in the source structure"]
            if n_unassigned
            else notes
        ),
        transformations=transformations,
    )


__all__ = [
    "STANDARDIZATION_VERSION",
    "STATUS_EMPTY",
    "STATUS_MULTI_FRAGMENT",
    "STATUS_NO_ORGANIC_FRAGMENT",
    "STATUS_OK",
    "STATUS_PARSE_FAILED",
    "STATUS_STANDARDIZE_FAILED",
    "StandardizationResult",
    "murcko_scaffold",
    "standardize_smiles",
]
