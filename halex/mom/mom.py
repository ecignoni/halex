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


def _orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, dim: int
) -> torch.Tensor:
    oom = _orbital_overlap_matrix(ovlp_12, c1, c2)
    proj = torch.sum(oom, dim=dim)
    return proj


def _batched_orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, dim: int
) -> torch.Tensor:
    oom = _batched_orbital_overlap_matrix(ovlp_12, c1, c2)
    proj = torch.sum(oom, dim=dim)
    return proj


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
    """computes the orbital overlap matrix

    This is the matrix used in MOM, eq 2.7 of
    https://pubs.acs.org/doi/pdf/10.1021/jp801738f
    """
    batching = _validate_arguments(ovlp_12, c1, c2)
    if batching:
        return _batched_orbital_overlap_matrix(ovlp_12, c1, c2)
    else:
        return _orbital_overlap_matrix(ovlp_12, c1, c2)


def orbital_projection(
    ovlp_12: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, which: str = "1over2"
) -> torch.Tensor:
    """projection of MO of 1 onto the MO space of 2

    This is the score computed in MOM, eq 2.8 of
    https://pubs.acs.org/doi/pdf/10.1021/jp801738f
    """
    batching = _validate_arguments(ovlp_12, c1, c2)
    dim = _validate_projection(which)
    if batching:
        return _batched_orbital_projection(ovlp_12, c1, c2, dim)
    else:
        return _orbital_projection(ovlp_12, c1, c2, dim)


# ============================================================================
# Utility functions
# ============================================================================


def indices_highest_orbital_projection(proj, n, skip_n):
    idx = torch.argsort(proj.abs()[:, skip_n:], dim=1, descending=True)[:, :n]
    idx = torch.sort(idx, dim=1).values + skip_n
    return idx
