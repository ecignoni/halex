import torch


def isqrtm(A):
    eva, eve = torch.linalg.eigh(A)
    return eve @ torch.diag(eva ** (-0.5)) @ eve.T


def sqrtm(A):
    eva, eve = torch.linalg.eigh(A)
    return eve @ torch.diag(eva**0.5) @ eve.T
