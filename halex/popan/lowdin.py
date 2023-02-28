import numpy as np
import torch
from ..operations import isqrtm


def lowdin_population(fock, ovlp, nelec_dict, ao_labels):
    ovlp_i12 = isqrtm(ovlp)
    fock_tilde = torch.einsum("ij,jk,kl->il", ovlp_i12, fock, ovlp_i12)
    eps, c_tilde = torch.linalg.eigh(fock_tilde)

    nmo = fock.shape[0]
    n_elec = int(
        sum(
            [
                nelec_dict[symb]
                for idx, symb in np.unique([(i, s) for i, s, _ in ao_labels], axis=0)
            ]
        )
    )
    mo_occ = torch.Tensor(
        [2 for i in range(n_elec // 2)] + [0 for i in range(nmo - n_elec // 2)]
    ).type(torch.int32)
    occidx = torch.Tensor([i for i, occ in enumerate(mo_occ) if occ != 0]).long()

    pop = 2 * torch.einsum("ia,ia->i", c_tilde[:, occidx], c_tilde[:, occidx].conj())
    atoms = np.unique([(idx, symb) for idx, symb, _ in ao_labels], axis=0)
    chg = torch.Tensor([nelec_dict[a] for (i, a) in atoms])
    for i, lbl in enumerate(ao_labels):
        atom_idx = int(lbl[0])
        chg[atom_idx] -= pop[i]

    return chg, pop


def orthogonal_lowdin_population(fock_orth, nelec_dict, ao_labels):
    eps, c_tilde = torch.linalg.eigh(fock_orth)

    nmo = fock_orth.shape[0]
    n_elec = int(
        sum(
            [
                nelec_dict[symb]
                for idx, symb in np.unique([(i, s) for i, s, _ in ao_labels], axis=0)
            ]
        )
    )
    mo_occ = torch.Tensor(
        [2 for i in range(n_elec // 2)] + [0 for i in range(nmo - n_elec // 2)]
    ).type(torch.int32)
    occidx = torch.Tensor([i for i, occ in enumerate(mo_occ) if occ != 0]).long()

    pop = 2 * torch.einsum("ia,ia->i", c_tilde[:, occidx], c_tilde[:, occidx].conj())
    atoms = np.unique([(idx, symb) for idx, symb, _ in ao_labels], axis=0)
    chg = torch.Tensor([nelec_dict[a] for (i, a) in atoms])
    for i, lbl in enumerate(ao_labels):
        atom_idx = int(lbl[0])
        chg[atom_idx] -= pop[i]

    return chg, pop


def batched_orthogonal_lowdin_population(focks_orth, nelec_dict, ao_labels):
    eps, c_tilde = torch.linalg.eigh(focks_orth)

    n_frames, nmo, _ = focks_orth.shape
    n_elec = int(
        sum(
            [
                nelec_dict[symb]
                for idx, symb in np.unique([(i, s) for i, s, _ in ao_labels], axis=0)
            ]
        )
    )
    mo_occ = torch.Tensor(
        [2 for i in range(n_elec // 2)] + [0 for i in range(nmo - n_elec // 2)]
    ).type(torch.int32)
    occidx = torch.Tensor([i for i, occ in enumerate(mo_occ) if occ != 0]).long()

    pop = 2 * torch.einsum(
        "fia,fia->fi", c_tilde[:, :, occidx], c_tilde[:, :, occidx].conj()
    )
    atoms = np.unique([(idx, symb) for idx, symb, _ in ao_labels], axis=0)
    chg = torch.Tensor([[nelec_dict[a] for (i, a) in atoms]]).repeat(n_frames, 1)
    for i, lbl in enumerate(ao_labels):
        atom_idx = int(lbl[0])
        chg[:, atom_idx] -= pop[:, i]

    return chg, pop


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
