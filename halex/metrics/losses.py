import torch
from torch_hamiltonians import (
    blocks_to_dense,
    decouple_blocks,
)
from .popan import batched_orthogonal_lowdin_population

# ============================================================================
# "Classic" loss functions
# ============================================================================


def _mean_squared_error_full(focks, pred_focks):
    # return torch.mean((fock - pred)**2) * Hartree**2
    n_frames, n_ao, _ = focks.shape
    loss = torch.empty(n_frames)
    for i in range(n_frames):
        loss[i] = torch.sum((focks[i] - pred_focks[i]) ** 2) / n_ao
    return torch.mean(loss)


def mean_squared_error_full(focks, pred_blocks, frames, orbs):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    return _mean_squared_error_full(focks, pred_focks)  # * Hartree**2


# ============================================================================
# Loss functions from derived quantities
# ============================================================================


def _mean_squared_error_eigvals(focks, pred_focks):
    evals = torch.linalg.eigvalsh(focks)
    pred_evals = torch.linalg.eigvalsh(pred_focks)
    return torch.mean((evals - pred_evals) ** 2)


def mean_squared_error_eigvals(focks, pred_blocks, frames, orbs):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    return _mean_squared_error_eigvals(focks, pred_focks)  # * Hartree**2


def _mean_squared_error_lowdinq(focks, pred_focks, nelec_dict, ao_labels, weights=None):
    lowdinq, _ = batched_orthogonal_lowdin_population(focks, nelec_dict, ao_labels)
    pred_lowdinq, _ = batched_orthogonal_lowdin_population(
        pred_focks, nelec_dict, ao_labels
    )
    loss = torch.mean((lowdinq - pred_lowdinq) ** 2, dim=0)
    if weights is not None:
        weights = weights / torch.sum(weights)
    else:
        n_charges = lowdinq.shape[1]
        weights = torch.ones(n_charges) / n_charges
    return torch.mean(loss * weights)


def mean_squared_error_lowdinq(
    focks, pred_blocks, frames, orbs, nelec_dict, ao_labels, weights=None
):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    return _mean_squared_error_lowdinq(
        focks, pred_focks, nelec_dict, ao_labels, weights=weights
    )


def _mean_squared_error_mocoeff(focks, pred_focks):
    eva, eve = torch.linalg.eigh(focks)
    eva_pred, eve_pred = torch.linalg.eigh(pred_focks)
    return torch.mean((eve**2 - eve_pred**2) ** 2)


def mean_squared_error_mocoeff(focks, pred_blocks, frames, orbs):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    return _mean_squared_error_mocoeff(focks, pred_focks)


# ============================================================================
# Composite loss functions
# ============================================================================


def mean_squared_error_full_eigvals(focks, pred_blocks, frames, orbs, a=1.0, b=1.0):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    loss_a = _mean_squared_error_full(focks, pred_focks)  # * Hartree**2
    loss_b = _mean_squared_error_eigvals(focks, pred_focks)  # * Hartree**2
    return a * loss_a + b * loss_b, loss_a, loss_b


def mean_squared_error_eigvals_mocoeff(focks, pred_blocks, frames, orbs, a=1.0, b=1.0):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    loss_a = _mean_squared_error_eigvals(focks, pred_focks)
    loss_b = _mean_squared_error_mocoeff(focks, pred_focks)
    return a * loss_a + b * loss_b, loss_a, loss_b


def mean_squared_error_eigvals_lowdinq(
    focks, pred_blocks, frames, orbs, nelec_dict, ao_labels, weights=None, a=1.0, b=1.0
):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    loss_a = _mean_squared_error_eigvals(focks, pred_focks)
    loss_b = _mean_squared_error_lowdinq(
        focks, pred_focks, nelec_dict, ao_labels, weights=weights
    )
    return a * loss_a + b * loss_b, loss_a, loss_b


def mean_squared_error_full_eigvals_lowdinq(
    focks,
    pred_blocks,
    frames,
    orbs,
    nelec_dict,
    ao_labels,
    weights=None,
    a=1.0,
    b=1.0,
    c=1.0,
):
    pred_focks = torch.stack(
        blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
    )
    loss_a = _mean_squared_error_full(focks, pred_focks)  # * Hartree**2
    loss_b = _mean_squared_error_eigvals(focks, pred_focks)  # * Hartree**2
    loss_c = _mean_squared_error_lowdinq(
        focks, pred_focks, nelec_dict, ao_labels, weights=weights
    )
    return a * loss_a + b * loss_b + c * loss_c, loss_a, loss_b, loss_c


# ============================================================================
# Aliases
# ============================================================================

mse_full = mean_squared_error_full
mse_eigvals = mean_squared_error_eigvals
mse_lowdinq = mean_squared_error_lowdinq
mse_mocoeff = mean_squared_error_mocoeff
mse_full_eigvals = mean_squared_error_full_eigvals
mse_full_lowdinq = mean_squared_error_eigvals_lowdinq
mse_full_eigvals_lowdinq = mean_squared_error_full_eigvals_lowdinq
mse_eigvals_mocoeff = mean_squared_error_eigvals_mocoeff
