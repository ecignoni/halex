import torch
from .utils import sqrtm


def charge_density_monopole(ovlp, natm, ao_labels, mo_coeff_a, mo_coeff_b):
    """computes the q_pq^A using Löwdin population analysis.

        q_pq^A = sum_(μ ϵ A) C'μp^(a) C'μq^(b)
        C' = S^(½) C

    Args:
        ovlp (torch.Tensor, (n_ao, n_ao)): AO overlap matrix (S).
        natm (int): number of atoms.
        ao_labels (list): list of tuples describing AOs.
                          same as calling mol.ao_labels(fmt=None) from pyscf.
                          tuple fmt: (atom_index: int, atom: str, ao_name: str, m_def: str)
        mo_coeff_a (torch.Tensor, (n_ao_a, n_mo_a)): MO coefficients matrix (C).
        mo_coeff_b (torch.Tensor, (n_ao_b, n_mo_b)): MO coefficients matrix (C).
    Returns:
        q (torch.Tensor, (natm, n_mo_a, n_mo_b)): charges from Löwdin population analysis.
    """
    ovlp_i12 = sqrtm(ovlp)
    coeff_orth_a = torch.matmul(ovlp_i12, mo_coeff_a)
    coeff_orth_b = torch.matmul(ovlp_i12, mo_coeff_b)
    nmo_a = coeff_orth_a.shape[1]
    nmo_b = coeff_orth_b.shape[1]
    q = torch.zeros((natm, nmo_a, nmo_b))
    for i, (atidx, *_) in enumerate(ao_labels):
        q[atidx] += torch.einsum("p,q->pq", coeff_orth_a[i], coeff_orth_b[i]).real
    return q
