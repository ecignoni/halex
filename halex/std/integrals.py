import torch

from .utils import isqrtm


def charge_density_monopole(ovlp, natm, ao_labels, mo_coeff_a, mo_coeff_b):
    ovlp_i12 = isqrtm(ovlp)
    coeff_orth_a = torch.matmul(ovlp_i12, mo_coeff_a)
    coeff_orth_b = torch.matmul(ovlp_i12, mo_coeff_b)
    nmo_a = coeff_orth_a.shape[1]
    nmo_b = coeff_orth_b.shape[1]
    q = torch.zeros((natm, nmo_a, nmo_b))
    for i, (atidx, *_) in enumerate(ao_labels):
        q[atidx] += torch.einsum("p,q->pq", coeff_orth_a[i], coeff_orth_b[i]).real
    return q
