from __future__ import annotations
import torch


def sqrtm(A: torch.Tensor) -> torch.Tensor:
    eva, eve = torch.linalg.eigh(A)
    idx = eva > 1e-15
    return eve[:, idx] @ torch.diag(eva[idx] ** 0.5) @ eve[:, idx].T


def isqrtm(A: torch.Tensor) -> torch.Tensor:
    eva, eve = torch.linalg.eigh(A)
    idx = eva > 1e-15
    return eve[:, idx] @ torch.diag(eva[idx] ** (-0.5)) @ eve[:, idx].T


def lowdin_orthogonalize(F: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    if F.ndim == 3:
        Ss_i12 = [isqrtm(s) for s in S]
        return torch.stack([s_i12 @ f @ s_i12 for f, s_i12 in zip(F, Ss_i12)])
    else:
        S_i12 = isqrtm(S)
        return S_i12 @ F @ S_i12
