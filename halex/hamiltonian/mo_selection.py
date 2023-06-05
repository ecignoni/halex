import numpy as np
import scipy as sp


def compute_saph(
    fock, over, frame, orbs, sel_types, n_core, orthogonality_threshold=1e-8
):
    """Computes symmetry-adapted projected Hamiltonian by projecting the
    key molecular orbitals onto a smaller basis than the full Hamiltonian basis.
    Assumes to be given the non-orthogonal Hamiltonian and the overlap matrix"""

    # first solves the non-orthogonal eigenvalue problem to get the target eigenvalues and eigenvectors
    l, U = sp.linalg.eigh(fock, over)

    # finds the selected basis indices for the given frame
    sel_basis = []
    sel_k = 0
    tot_core = 0
    for s in frame.symbols:
        sel_basis.append(np.asarray(sel_types[s], dtype=int) + sel_k)
        tot_core += n_core[s]
        sel_k += len(orbs[s])

    sel_basis = np.concatenate(sel_basis)

    # first guess at MO selection - drop core states and pick the size
    # of the selected basis plus some buffer which we use to sort out
    # orthogonality problems
    sel_mo = np.arange(tot_core, tot_core + len(sel_basis) + 8)

    # these are the coefficients projected on the selected basis
    V = over[sel_basis] @ U[:, sel_mo]
    u, s, vt = sp.linalg.svd(V, full_matrices=False)

    # determines the relevant symmetry-adapted subspace
    ovt = vt.copy()
    osel_mo = []
    selected = []
    # strategy is that we do Gram-Schmidt orthogonalization without renormalizing.
    # when a MO cannot be described because it is fully linearly dependent on
    # already selected MOs, it skips
    for k in range(vt.shape[1]):
        if (ovt[:, k] @ ovt[:, k]) < orthogonality_threshold:
            continue
        selected.append(k)
        osel_mo.append(sel_mo[k])
        ovt[:, k] /= np.sqrt(ovt[:, k] @ ovt[:, k])
        for j in range(k + 1, vt.shape[1]):
            ovt[:, j] -= ovt[:, k] * (ovt[:, j] @ ovt[:, k])

    sel_mo = np.asarray(osel_mo[: len(sel_basis)], dtype=int)

    # now we use the selected MOs to build a SAPH matrix with the same eigenvalues
    # as the original one
    V = over[sel_basis] @ U[:, sel_mo]
    u, s, vt = sp.linalg.svd(V)
    o_V = u @ vt
    return o_V @ np.diag(l[sel_mo]) @ o_V.T, selected
