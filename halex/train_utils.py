from __future__ import annotations
from typing import Tuple, Dict, Any, List

import os
import numpy as np
from tqdm import tqdm

import equistore
from equistore import TensorMap

from .decomposition import EquivariantPCA
from .rotations import ClebschGordanReal
from .dataset import SCFData, BatchedMemoryDataset
from .utils import tensormap_as_torch, load_cross_ovlps
from .hamiltonian import (
    compute_ham_features,
    drop_unused_features,
    drop_noncore_features,
)
from .operations import unorthogonalize_coeff
from .mom import (
    mom_orbital_projection,
    pmom_orbital_projection,
    indices_highest_orbital_projection,
)

import torch


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
            feats, targs_coupled=small_basis_data.focks_orth_tmap_coupled
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


def batched_dataset_for_a_single_molecule(
    scf_datasets: Tuple[SCFData, SCFData],
    feats: List[TensorMap],
    nelec_dict: Dict[str, float],
    batch_size: int,
    lowdin_charges_by_MO: bool = False,
    core_feats: List[TensorMap] = None,
    mo_indices=None,
    lowdin_mo_indices=None,
    ignore_heavy_1s: bool = False,
) -> BatchedMemoryDataset:
    """
    Create a BatchedMemoryDataset (which is what our models expect)

    lowdin_charges_by_MO: whether to use total lowdin charges (False)
                          or lowdin charges partitioned by MO (True)
    core_feats: optional features used to learn the core elements of
                the Fock matrix.
    """
    small_basis, big_basis = scf_datasets

    frames = small_basis.frames

    # get the number of core elements as the number of atoms that are
    # not hydrogens. Choose on the basis of the first frame
    if ignore_heavy_1s:
        ncore = _number_of_heavy_elements(big_basis.frames[0])
    else:
        ncore = 0

    # truncate the big basis MO energies.
    # If indices are present, use them
    mo_energy = big_basis.mo_energy[:, ncore:]

    if mo_indices is None:
        mo_energy = mo_energy[:, : small_basis.mo_energy.shape[1] - ncore]
    else:
        mo_energy = torch.take(mo_energy, mo_indices)

    # no need to truncate here as they refer to _occupied_ MOs
    lowdin_charges = (
        big_basis.lowdin_charges_byMO[:, ncore:]
        if lowdin_charges_by_MO
        else big_basis.lowdin_charges
    )

    # orbitals in the small basis (because we predict a small basis Fock)
    orbs = small_basis.orbs
    if ignore_heavy_1s:
        orbs = _drop_heavy_1s_from_orbs(orbs)

    # ao labels in the small basis
    ao_labels = small_basis.ao_labels
    if ignore_heavy_1s:
        ao_labels = _drop_heavy_1s_from_ao_labels(ao_labels)

    if core_feats is None:
        return BatchedMemoryDataset(
            len(frames),
            feats,
            frames,
            mo_energy,
            lowdin_charges,
            ao_labels=ao_labels,
            orbs=orbs,
            nelec_dict=nelec_dict,
            batch_size=batch_size,
            lowdin_mo_indices=lowdin_mo_indices,
        )
    else:
        return BatchedMemoryDataset(
            len(frames),
            feats,
            core_feats,
            frames,
            mo_energy,
            lowdin_charges,
            ao_labels=ao_labels,
            orbs=orbs,
            nelec_dict=nelec_dict,
            batch_size=batch_size,
            lowdin_mo_indices=lowdin_mo_indices,
        )


def _number_of_heavy_elements(frame):
    return sum(frame.numbers != 1)


def _drop_heavy_1s_from_orbs(orbs):
    new_orbs = dict()
    for key in orbs.keys():
        if key == 1:
            new_orbs[key] = orbs[key]
        else:
            new_orbs[key] = list()
            for nlm in orbs[key]:
                if tuple(nlm) == (1, 0, 0):
                    pass
                else:
                    new_orbs[key].append(nlm)
    return new_orbs


def _drop_heavy_1s_from_ao_labels(ao_labels):
    new_ao_labels = []
    for lbl in ao_labels:
        if lbl[1] == "H":
            new_ao_labels.append(lbl)
        else:
            if tuple(lbl[2]) != (1, 0, 0):
                new_ao_labels.append(lbl)
    return new_ao_labels


def baselined_batched_dataset_for_a_single_molecule(
    scf_datasets: Tuple[SCFData, SCFData],
    feats: List[TensorMap],
    nelec_dict: Dict[str, float],
    batch_size: int,
    baseline_focks: torch.Tensor,
    lowdin_charges_by_MO: bool = False,
    core_feats: List[TensorMap] = None,
    mo_indices=None,
    lowdin_mo_indices=None,
    ignore_heavy_1s: bool = False,
) -> BatchedMemoryDataset:
    """
    Create a BatchedMemoryDataset (which is what our models expect)

    lowdin_charges_by_MO: whether to use total lowdin charges (False)
                          or lowdin charges partitioned by MO (True)
    core_feats: optional features used to learn the core elements of
                the Fock matrix.
    """
    small_basis, big_basis = scf_datasets

    frames = small_basis.frames

    # get the number of core elements as the number of atoms that are
    # not hydrogens. Choose on the basis of the first frame
    mo_energy = big_basis.mo_energy
    if mo_indices is None:
        mo_energy = mo_energy[:, : small_basis.mo_energy.shape[1]]
    else:
        mo_energy = torch.take(mo_energy, mo_indices)

    # no need to truncate here as they refer to _occupied_ MOs
    lowdin_charges = (
        big_basis.lowdin_charges_byMO
        if lowdin_charges_by_MO
        else big_basis.lowdin_charges
    )

    # orbitals in the small basis (because we predict a small basis Fock)
    orbs = small_basis.orbs

    # ao labels in the small basis
    ao_labels = small_basis.ao_labels

    if core_feats is None:
        return BatchedMemoryDataset(
            len(frames),
            feats,
            frames,
            mo_energy,
            lowdin_charges,
            baseline_focks,
            ao_labels=ao_labels,
            orbs=orbs,
            nelec_dict=nelec_dict,
            batch_size=batch_size,
            lowdin_mo_indices=lowdin_mo_indices,
        )
    else:
        return BatchedMemoryDataset(
            len(frames),
            feats,
            core_feats,
            frames,
            mo_energy,
            lowdin_charges,
            baseline_focks,
            ao_labels=ao_labels,
            orbs=orbs,
            nelec_dict=nelec_dict,
            batch_size=batch_size,
            lowdin_mo_indices=lowdin_mo_indices,
        )


def coupled_fock_matrix_from_multiple_molecules(
    multimol_scf_datasets: List[Tuple[SCFData, SCFData]]
) -> TensorMap:
    """
    Get and join together the Fock matrices in the coupled
    angular momentum basis for multiple molecules
    """
    to_couple = []
    if len(multimol_scf_datasets) > 1:
        for small_basis, _ in multimol_scf_datasets:
            to_couple.append(small_basis.focks_orth_tmap_coupled)
            # to_couple.append(shift_structure_by_n(smallb.focks_orth_tmap_coupled, n=n))
            # n += smallb.n_frames
        fock_coupled = equistore.join(to_couple, axis="samples")
    else:
        fock_coupled = list(multimol_scf_datasets)[0][0].focks_orth_tmap_coupled
    return fock_coupled


def indices_from_MOM(cross_ovlp_paths, scf_datasets):
    indices = {}
    projections = []
    for path, (mol, (sb, bb)) in zip(cross_ovlp_paths, scf_datasets.items()):
        cross_ovlps = load_cross_ovlps(
            path,
            frames=sb.frames,
            orbs_sb=sb.orbs,
            orbs_bb=bb.orbs,
            indices=sb.indices,
        )
        c_sb = unorthogonalize_coeff(sb.ovlps, sb.mo_coeff_orth)
        c_bb = unorthogonalize_coeff(bb.ovlps, bb.mo_coeff_orth)
        proj = mom_orbital_projection(cross_ovlps, c_sb, c_bb, which="2over1")
        projections.append(proj)
        nocc = sum(sb.mo_occ == 2).item()
        nvir = sum(sb.mo_occ == 0).item()
        mo_vir_idx = indices_highest_orbital_projection(proj, n=nvir, skip_n=nocc)
        mo_occ = torch.repeat_interleave(
            torch.arange(nocc)[None, :], mo_vir_idx.shape[0], dim=0
        )
        selected = torch.column_stack([mo_occ, mo_vir_idx])
        indices[mol] = selected
    return indices, projections


def indices_from_PMOM(cross_ovlp_paths, scf_datasets):
    indices = {}
    projections = []
    for path, (mol, (sb, bb)) in zip(cross_ovlp_paths, scf_datasets.items()):
        cross_ovlps = load_cross_ovlps(
            path,
            frames=sb.frames,
            orbs_sb=sb.orbs,
            orbs_bb=bb.orbs,
            indices=sb.indices,
        )
        c_sb = unorthogonalize_coeff(sb.ovlps, sb.mo_coeff_orth)
        c_bb = unorthogonalize_coeff(bb.ovlps, bb.mo_coeff_orth)
        proj = pmom_orbital_projection(cross_ovlps, c_sb, c_bb, which="2over1")
        projections.append(proj)
        nocc = sum(sb.mo_occ == 2).item()
        nvir = sum(sb.mo_occ == 0).item()
        mo_vir_idx = indices_highest_orbital_projection(proj, n=nvir, skip_n=nocc)
        mo_occ = torch.repeat_interleave(
            torch.arange(nocc)[None, :], mo_vir_idx.shape[0], dim=0
        )
        selected = torch.column_stack([mo_occ, mo_vir_idx])
        indices[mol] = selected
    return indices, projections
