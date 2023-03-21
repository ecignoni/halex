from __future__ import annotations
from typing import Tuple, Dict, List, Any

import numpy as np
import torch
from ..operations import isqrtm


# ============================================================================
# Utilities
# ============================================================================


def _lowdin_orthogonalize(
    fock: torch.Tensor, ovlp: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ovlp_i12 = isqrtm(ovlp)
    return torch.einsum("ij,jk,kl->il", ovlp_i12, fock, ovlp_i12)


def _get_n_elec(nelec_dict: Dict[int, float], ao_labels: List[Tuple[int, Any]]) -> int:
    return int(
        sum(
            [
                nelec_dict[symb]
                for idx, symb in np.unique([(i, s) for i, s, _ in ao_labels], axis=0)
            ]
        )
    )


def _get_mo_occ(nmo: int, n_elec: int) -> torch.Tensor:
    return torch.Tensor(
        [2 for i in range(n_elec // 2)] + [0 for i in range(nmo - n_elec // 2)]
    ).long()


def _get_occidx(mo_occ: torch.Tensor) -> torch.Tensor:
    return torch.Tensor([i for i, occ in enumerate(mo_occ) if occ != 0]).long()


def _get_atom_charges(
    nelec_dict: Dict[int, float], ao_labels: List[Tuple[int, Any]]
) -> torch.Tensor:
    atoms = np.unique([(idx, symb) for idx, symb, _ in ao_labels], axis=0)
    return torch.DoubleTensor([nelec_dict[a] for (_, a) in atoms])


# ============================================================================
# Utilities
# ============================================================================


def lowdin_population(
    fock: torch.Tensor,
    ovlp: torch.Tensor,
    nelec_dict: Dict[int, float],
    ao_labels: List[Tuple[int, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Löwdin population analysis"""
    fock_tilde = _lowdin_orthogonalize(fock, ovlp)
    eps, c_tilde = torch.linalg.eigh(fock_tilde)

    nmo = fock.shape[0]
    n_elec = _get_n_elec(nelec_dict, ao_labels)

    mo_occ = _get_mo_occ(nmo, n_elec)
    occidx = _get_occidx(mo_occ)

    pop = 2 * torch.einsum("ia,ia->i", c_tilde[:, occidx], c_tilde[:, occidx].conj())
    chg = _get_atom_charges(nelec_dict, ao_labels)
    for i, (iat, *_) in enumerate(ao_labels):
        chg[iat] -= pop[i]

    return chg, pop


def orthogonal_lowdin_population(
    fock_orth: torch.Tensor,
    nelec_dict: Dict[int, float],
    ao_labels: List[Tuple[int, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Löwdin population analysis"""
    eps, c_tilde = torch.linalg.eigh(fock_orth)

    nmo = fock_orth.shape[0]
    n_elec = _get_n_elec(nelec_dict, ao_labels)

    mo_occ = _get_mo_occ(nmo, n_elec)
    occidx = _get_occidx(mo_occ)

    pop = 2 * torch.einsum("ia,ia->i", c_tilde[:, occidx], c_tilde[:, occidx].conj())
    chg = _get_atom_charges(nelec_dict, ao_labels)
    for i, (iat, *_) in enumerate(ao_labels):
        chg[iat] -= pop[i]

    return chg, pop


def batched_orthogonal_lowdin_population(
    focks_orth: torch.Tensor,
    nelec_dict: Dict[int, float],
    ao_labels: List[Tuple[int, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Löwdin population analysis"""
    eps, c_tilde = torch.linalg.eigh(focks_orth)

    n_frames, nmo, _ = focks_orth.shape
    n_elec = _get_n_elec(nelec_dict, ao_labels)

    mo_occ = _get_mo_occ(nmo, n_elec)
    occidx = _get_occidx(mo_occ)

    pop = 2 * torch.einsum(
        "fia,fia->fi", c_tilde[:, :, occidx], c_tilde[:, :, occidx].conj()
    )
    chg = _get_atom_charges(nelec_dict, ao_labels)
    chg = chg.repeat(n_frames, 1)
    for i, (iat, *_) in enumerate(ao_labels):
        chg[:, iat] -= pop[:, i]

    return chg, pop


def orthogonal_lowdinbyMO_population(
    fock_orth: torch.Tensor,
    nelec_dict: Dict[int, float],
    ao_labels: List[Tuple[int, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Löwdin population analysis"""
    eps, c_tilde = torch.linalg.eigh(fock_orth)
    nmo = fock_orth.shape[0]

    n_elec = _get_n_elec(nelec_dict, ao_labels)

    mo_occ = _get_mo_occ(nmo, n_elec)
    occidx = _get_occidx(mo_occ)

    pop = 2 * torch.einsum("mi,mi->im", c_tilde[:, occidx], c_tilde[:, occidx].conj())
    nocc = len(occidx)
    atom_charges = _get_atom_charges(nelec_dict, ao_labels)
    natm = len(atom_charges)
    chg = torch.zeros((nocc, natm))
    chg[:] = atom_charges
    for i, (iat, *_) in enumerate(ao_labels):
        chg[:, iat] -= pop[:, i]

    return chg


def batched_orthogonal_lowdinbyMO_population(
    focks_orth: torch.Tensor,
    nelec_dict: Dict[int, float],
    ao_labels: List[Tuple[int, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Löwdin population analysis"""
    eps, c_tilde = torch.linalg.eigh(focks_orth)

    n_frames, nmo, _ = focks_orth.shape
    n_elec = _get_n_elec(nelec_dict, ao_labels)

    mo_occ = _get_mo_occ(nmo, n_elec)
    occidx = _get_occidx(mo_occ)

    pop = 2 * torch.einsum(
        "fmi,fmi->fim", c_tilde[:, :, occidx], c_tilde[:, :, occidx].conj()
    )
    nocc = len(occidx)
    atom_charges = _get_atom_charges(nelec_dict, ao_labels)
    natm = len(atom_charges)
    chg = torch.zeros((n_frames, nocc, natm))
    chg[:, :] = atom_charges
    for i, (iat, *_) in enumerate(ao_labels):
        chg[:, :, iat] -= pop[:, :, i]

    return chg


# if __name__ == '__main__':
#
#     def check_lowdin():
#         nelec_dict = {"H": 1.0, "O": 8.0}
#         out = np.load("data/water-hamiltonian/water_out_pyscf.npz")
#         fock = out["fock"]
#         ovlp = out["ovlp"]
#         ao_labels = [
#             (int(lbl.split()[0]), lbl.split()[1], lbl.split()[2])
#             for lbl in out["ao_labels"]
#         ]
#         ref_chg = np.array([-0.0903, -0.0111, 0.1014])
#
#         to_torch = lambda *arr: (torch.from_numpy(a).type(torch.float64) for a in arr)
#         fock, ovlp, ref_chg = to_torch(fock, ovlp, ref_chg)
#         chg, pop = lowdin_population(fock, ovlp, nelec_dict, ao_labels)
#
#         np.testing.assert_allclose(chg, ref_chg, rtol=1e-3)
#
#     def check_orthogonal_lowdin():
#         nelec_dict = {"H": 1.0, "O": 8.0}
#         out = np.load("data/water-hamiltonian/water_out_pyscf.npz")
#         fock = out["fock"]
#         ovlp = out["ovlp"]
#         ao_labels = [
#             (int(lbl.split()[0]), lbl.split()[1], lbl.split()[2])
#             for lbl in out["ao_labels"]
#         ]
#         ref_chg = np.array([-0.0903, -0.0111, 0.1014])
#
#         to_torch = lambda *arr: (torch.from_numpy(a).type(torch.float64) for a in arr)
#         fock, ovlp, ref_chg = to_torch(fock, ovlp, ref_chg)
#         ovlp_i12 = lowdin_transformation(ovlp)
#         fock = ovlp_i12 @ fock @ ovlp_i12
#         chg, pop = orthogonal_lowdin_population(fock, nelec_dict, ao_labels)
#
#         np.testing.assert_allclose(chg, ref_chg, rtol=1e-3)
#
#     def check_batched_orthogonal_lowdin():
#         nelec_dict = {"H": 1.0, "O": 8.0}
#         out = np.load("data/water-hamiltonian/water_out_pyscf.npz")
#         fock = out["fock"]
#         ovlp = out["ovlp"]
#         ao_labels = [
#             (int(lbl.split()[0]), lbl.split()[1], lbl.split()[2])
#             for lbl in out["ao_labels"]
#         ]
#         ref_chg = np.array([-0.0903, -0.0111, 0.1014])
#
#         to_torch = lambda *arr: (torch.from_numpy(a).type(torch.float64) for a in arr)
#         fock, ovlp, ref_chg = to_torch(fock, ovlp, ref_chg)
#         ovlp_i12 = lowdin_transformation(ovlp)
#         fock = ovlp_i12 @ fock @ ovlp_i12
#         chg, pop = batched_orthogonal_lowdin_population(fock[None, :], nelec_dict, ao_labels)
#
#         np.testing.assert_allclose(chg[0], ref_chg, rtol=1e-3)
