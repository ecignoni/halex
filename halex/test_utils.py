from __future__ import annotations
from typing import List, Callable, Any

from .rotations.rotations import rotation_matrix, wigner_d_real
import warnings
import numpy as np
import torch

Atoms = Any


def test_rotation_equivariance(
    frames: List[Atoms],
    test_fn: Callable,
    raise_exception: bool = True,
    rtol: float = 1e-14,
    key_l_name: str = "spherical_harmonics_l",
) -> None:
    """rotation equivariance test

    Tests for the rotation equivariance of the function f (test_fn)
    when applied to one or more structures (frames)

        R·f(A) = f(R·A)

    where R is a rotation operator, A is a structure, f is the
    tested function.

    Args:
        frames: list of ase.io.Atoms objects
        test_fn: function computing f(A) for a single frame
                 should return a TensorMap object
        raise_exception: whether to raise an exception if the
                         equivariance is not fullfilled
        rtol: tolerance in the norm of f(R·A) - R·f(A)
        key_l_name: name of the angular l channel in the key of
                    the tensormap
    """
    for frame in frames:
        # setup the wigner rotation matrix
        alpha = np.random.uniform(0, 2 * np.pi)
        beta = np.random.uniform(0, np.pi)
        gamma = np.random.uniform(0, 2 * np.pi)
        R = rotation_matrix(alpha, beta, gamma).T

        A = frame.copy()
        RA = frame.copy()
        RA.positions = RA.positions @ R
        RA.cell = RA.cell @ R

        f_A = test_fn(A)
        f_RA = test_fn(RA)

        for (key_A, block_A), (key_RA, block_RA) in zip(f_A, f_RA):
            assert key_A == key_RA

            # rotation of f(A)
            l = int(key_A[key_l_name])  # noqa
            D = torch.from_numpy(wigner_d_real(l, alpha, beta, gamma))
            Rf_A = torch.einsum("nm,smp->snp", D, block_A.values)

            # compare with f(RA)
            norm = torch.linalg.norm(Rf_A - block_RA.values)
            if norm > rtol:
                if raise_exception:
                    raise RuntimeError(f"mismatch for key={key_A}, norm={norm}")
                else:
                    warnings.warn(f"mismatch for key={key_A}, norm={norm}")
