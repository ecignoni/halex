import torch

# ============================================================================
# Base Functions
# ============================================================================


def _batched_orbital_overlap_matrix(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("fmi,fmn,fnj->fij", c1.conj(), ovlp_12, c2)


def _orbital_overlap_matrix(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("mi,mn,nj->ij", c1.conj(), ovlp_12, c2)


def _mom_orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, dim: int
) -> torch.Tensor:
    oom = _orbital_overlap_matrix(ovlp_12, c1, c2)
    proj = torch.sum(oom, dim=dim)
    return proj


def _batched_mom_orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, dim: int
) -> torch.Tensor:
    oom = _batched_orbital_overlap_matrix(ovlp_12, c1, c2)
    proj = torch.sum(oom, dim=dim)
    return proj


def _pmom_orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, dim: int
) -> torch.Tensor:
    oom = _orbital_overlap_matrix(ovlp_12, c1, c2)
    proj = torch.sum(oom**2, dim=dim) ** 0.5
    return proj


def _batched_pmom_orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, dim: int
) -> torch.Tensor:
    oom = _batched_orbital_overlap_matrix(ovlp_12, c1, c2)
    proj = torch.sum(oom**2, dim=dim) ** 0.5
    return proj


# ============================================================================
# Interface helpers
# ============================================================================


def _validate_arguments(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor
) -> bool:
    if ovlp_12.ndim == 2 and c1.ndim == 2 and c2.ndim == 2:
        return False
    elif ovlp_12.ndim == 3 and c1.ndim == 3 and c2.ndim == 3:
        return True
    else:
        raise ValueError(
            "Input with wrong dimensions, got "
            f"ovlp_12.ndim = {ovlp_12.ndim} c1.ndim = {c1.ndim}"
            f" c2.ndim = {c2.ndim}"
        )


def _validate_projection(which: str) -> int:
    if which == "1over2":
        return -1
    elif which == "2over1":
        return -2
    else:
        raise ValueError(f"Input {which} not recognized.")


# ============================================================================
# Interface Functions
# ============================================================================


def orbital_overlap_matrix(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor
) -> torch.Tensor:
    r"""computes the orbital overlap matrix

    This is the matrix used in MOM, eq 2.7 of
    https://pubs.acs.org/doi/pdf/10.1021/jp801738f

        O_{ij} = C1_{μi} S12_{μλ} C2_{λj}

    where C1 is the (n_ao1, n_mo1) matrix of MO coefficients,
    the same holds for C2 (although, in general, they are different),
    and S12 is the (cross) overlap between C1 and C2.
    Repeating indices imply a summation.

    Args:
        ovlp_12: cross overlap between 1 and 2,
                 shape (n_ao1, n_ao2) or (n_frames, n_ao1, n_ao2)
        c1: MO coefficients (first set)
            shape (n_ao1, n_mo1) or (n_frames, n_ao1, n_mo1)
        c2: MO coefficients (second set)
            shape (n_ao2, n_mo2) or (n_frames, n_ao2, n_mo2)
    Returns:
        O: orbital overlap matrix
           shape (n_mo1, n_mo2) or (n_frames, n_mo1, n_mo2)
    """
    batching = _validate_arguments(ovlp_12, c1, c2)
    if batching:
        return _batched_orbital_overlap_matrix(ovlp_12, c1, c2)
    else:
        return _orbital_overlap_matrix(ovlp_12, c1, c2)


def mom_orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, which: str = "1over2"
) -> torch.Tensor:
    r"""projection of MO of 1 onto the MO space of 2

    This is the score computed in MOM, eq 2.8 of
    https://pubs.acs.org/doi/pdf/10.1021/jp801738f

        p_j = Σ_i O_{ij}

    Note that this is *not* a real projection, and can suffer
    from several instabilities. The projection also changes when
    the orbitals in C1 or C2 are rotated with a unitary matrix.

    Args:
        ovlp_12: cross overlap between 1 and 2,
                 shape (n_ao1, n_ao2) or (n_frames, n_ao1, n_ao2)
        c1: MO coefficients (first set)
            shape (n_ao1, n_mo1) or (n_frames, n_ao1, n_mo1)
        c2: MO coefficients (second set)
            shape (n_ao2, n_mo2) or (n_frames, n_ao2, n_mo2)
        str: which projection to perform
             "1over2" projects the MOs C1 onto C2
             "2over1" projects the MOs C2 onto C1
    Returns:
        p_j: MOM projection
           shape (n_mo1) or (n_frames, n_mo1) (str = "1over2")
           shape (n_mo2) or (n_frames, n_mo2) (str = "2over1")
    """
    batching = _validate_arguments(ovlp_12, c1, c2)
    dim = _validate_projection(which)
    if batching:
        return _batched_mom_orbital_projection(ovlp_12, c1, c2, dim)
    else:
        return _mom_orbital_projection(ovlp_12, c1, c2, dim)


def pmom_orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, which: str = "1over2"
) -> torch.Tensor:
    r"""projection of MO of 1 onto the MO space of 2

    This is the score computed in e.g. IMOM, eq 1 of
    https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.7b00994

        p_j = (Σ_i O_{ij}²)^{½}

    This is a real projection. It can be understood as comparing
    MO j with its projection onto the space of the other set of MOs.
    This orbital projection does not change if
    the orbitals over which the projection is performed
    are rotated with a unitary matrix.

    Args:
        ovlp_12: cross overlap between 1 and 2,
                 shape (n_ao1, n_ao2) or (n_frames, n_ao1, n_ao2)
        c1: MO coefficients (first set)
            shape (n_ao1, n_mo1) or (n_frames, n_ao1, n_mo1)
        c2: MO coefficients (second set)
            shape (n_ao2, n_mo2) or (n_frames, n_ao2, n_mo2)
        str: which projection to perform
             "1over2" projects the MOs C1 onto C2
             "2over1" projects the MOs C2 onto C1
    Returns:
        p_j: MOM projection
           shape (n_mo1) or (n_frames, n_mo1) (str = "1over2")
           shape (n_mo2) or (n_frames, n_mo2) (str = "2over1")
    """
    batching = _validate_arguments(ovlp_12, c1, c2)
    dim = _validate_projection(which)
    if batching:
        return _batched_pmom_orbital_projection(ovlp_12, c1, c2, dim)
    else:
        return _pmom_orbital_projection(ovlp_12, c1, c2, dim)


# ============================================================================
# Utility functions
# ============================================================================


def indices_highest_orbital_projection(proj, n, skip_n):
    """

    Args:
        ovlp_12: cross overlap between 1 and 2,
                 shape (n_ao1, n_ao2) or (n_frames, n_ao1, n_ao2)
        c1: MO coefficients (first set)
            shape (n_ao1, n_mo1) or (n_frames, n_ao1, n_mo1)
        c2: MO coefficients (second set)
            shape (n_ao2, n_mo2) or (n_frames, n_ao2, n_mo2)
    Returns:
        O: orbital overlap matrix
           shape (n_mo1, n_mo2) or (n_frames, n_mo1, n_mo2)
    """
    idx = torch.argsort(proj.abs()[:, skip_n:], dim=1, descending=True)[:, :n]
    idx = torch.sort(idx, dim=1).values + skip_n
    return idx
