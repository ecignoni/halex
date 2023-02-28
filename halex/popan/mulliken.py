import numpy as np
import torch
from ..operations import isqrtm


def mulliken_population(fock, ovlp, nelec_dict, ao_labels):
    ovlp_i12 = isqrtm(ovlp)
    fock_tilde = torch.einsum("ij,jk,kl->il", ovlp_i12, fock, ovlp_i12)
    eps, c_tilde = torch.linalg.eigh(fock_tilde)
    c = torch.matmul(ovlp_i12, c_tilde)

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
    dm = 2 * torch.einsum("ia,ja->ij", c[:, occidx], c[:, occidx].conj())

    pop = torch.einsum("ij,ji->i", dm, ovlp)
    atoms = np.unique([(idx, symb) for idx, symb, _ in ao_labels], axis=0)
    chg = torch.Tensor([nelec_dict[a] for (i, a) in atoms])
    for i, lbl in enumerate(ao_labels):
        atom_idx = int(lbl[0])
        chg[atom_idx] -= pop[i]

    return chg, pop


# if __name__ == "__main__":
#
#     def check_mulliken():
#         nelec_dict = {"H": 1.0, "O": 8.0}
#         out = np.load("data/water-hamiltonian/water_out_pyscf.npz")
#         fock = out["fock"]
#         ovlp = out["ovlp"]
#         ao_labels = [
#             (int(lbl.split()[0]), lbl.split()[1], lbl.split()[2])
#             for lbl in out["ao_labels"]
#         ]
#         ref_pop, ref_chg = out["pop"], out["chg"]
#
#         to_torch = lambda *arr: (torch.from_numpy(a).type(torch.float64) for a in arr)
#         fock, ovlp, ref_pop, ref_chg = to_torch(fock, ovlp, ref_pop, ref_chg)
#         chg, pop = mulliken_population(fock, ovlp, nelec_dict, ao_labels)
#
#         np.testing.assert_allclose(chg, ref_chg, rtol=1e-5)
#         np.testing.assert_allclose(pop, ref_pop, rtol=1e-5)
