from __future__ import annotations
from typing import Tuple, Dict, Any, List

import os
import numpy as np
from tqdm import tqdm

from equistore import TensorMap

from .decomposition import EquivariantPCA
from .rotations import ClebschGordanReal
from .dataset import SCFData
from .utils import tensormap_as_torch
from .hamiltonian import (
    compute_ham_features,
    drop_unused_features,
    drop_noncore_features,
)


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


def compute_features(
    datasets: Dict[str, Tuple[SCFData, SCFData]],
    rascal_hypers: Dict[str, Any],
    cg: ClebschGordanReal,
    lcut: int,
    epca: EquivariantPCA = None,
    core_only: bool = False,
) -> List[TensorMap]:
    """Computes the Hamiltonian features

    epca: if given, transforms the features using the fitted EquivariantPCA object
    core_only: if true, only retains the features used to learn core hamiltonian
               elements
    """
    calc_feats = lambda dataset: tensormap_as_torch(  # noqa
        compute_ham_features(rascal_hypers, frames=dataset.frames, cg=cg, lcut=lcut)
    )

    feats_list = []
    # n = 0
    for small_basis_data, big_basis_data in tqdm(datasets.values()):
        feats = calc_feats(small_basis_data)
        # Only retain "core" features if requested
        if core_only:
            feats = drop_noncore_features(feats)

        # Drop every other feature block that is not used to
        # learn our target (e.g., wrong symmetries)
        feats = drop_unused_features(
            feats, targs_coupled=big_basis_data.focks_orth_tmap_coupled
        )

        # Possibily transform the features with a pretrained
        # EquivariantPCA is available (save memory)
        if epca is not None:
            feats = epca.transform(feats)

        # feats = shift_structure_by_n(feats, n=n)
        # n += small_basis_data.n_frames
        feats_list.append(feats)

    # feats = equistore.join(feats_list, axis="samples")
    return feats_list
