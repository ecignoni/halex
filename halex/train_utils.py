from __future__ import annotations
from typing import Tuple

import os
import numpy as np
from .rotations import ClebschGordanReal
from .dataset import SCFData


def load_molecule_scf_datasets(
    coords_path: str,
    small_basis_path: str,
    big_basis_path: str,
    train_indices: np.ndarray,
    cg: ClebschGordanReal,
) -> Tuple[SCFData, SCFData]:
    r"""
    Load the SCFData objects storing data for a single molecule,
    in both a small basis and a big basis

    It is assumed that fock and overlap matrices are stored in a
    folder in .npy format, under the names focks.npy and ovlps.npy.

    focks.npy should contain the nonorthogonal fock matrices.
    ovlps.npy should contain the (possibly unnormalized) overlap matrices.

    Furthermore, a file orbs.json should contain the definition of the
    AO basis. For example, a minimal basis for elements H and C could be:

            n  l  m
    {"C": [[1, 0, 0], [2, 0, 0], [2, 1, 1], [2, 1, -1], [2, 1, 0]], "H": [[1, 0, 0]]}

    e.g., everything is computed in a spherical basis. The AO basis is understood
    to follow the PySCF ordering.
    """
    # small basis
    sb_focks = os.path.join(small_basis_path, "focks.npy")
    sb_ovlps = os.path.join(small_basis_path, "ovlps.npy")
    sb_orbs = os.path.join(small_basis_path, "orbs.json")
    sb_data = SCFData(
        frames=coords_path,
        focks=sb_focks,
        ovlps=sb_ovlps,
        orbs=sb_orbs,
        cg=cg,
        indices=train_indices,
    )

    # big basis
    bb_focks = os.path.join(big_basis_path, "focks.npy")
    bb_ovlps = os.path.join(big_basis_path, "ovlps.npy")
    bb_orbs = os.path.join(big_basis_path, "orbs.json")
    bb_data = SCFData(
        frames=coords_path,
        focks=bb_focks,
        ovlps=bb_ovlps,
        orbs=bb_orbs,
        cg=cg,
        indices=train_indices,
    )

    return sb_data, bb_data
