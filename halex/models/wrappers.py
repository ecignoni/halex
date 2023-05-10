from __future__ import annotations
from typing import Dict, List, Any, Tuple

from collections import defaultdict
from .tmap_models import RidgeModel
from ..popan import (
    batched_orthogonal_lowdin_population,
    batched_orthogonal_lowdinbyMO_population,
)
from ..hamiltonian import blocks_to_dense, decouple_blocks

import numpy as np
import torch

from tqdm import tqdm

from equistore import TensorMap

Dataset = Any
Atoms = Any


def _predict_focks_vectorized(pred_blocks, frames, orbs):
    """predicted tensormap --> predicted dense matrices

    This is a vectorized (faster evaluation, faster backpropagation)
    version of the prediction. Only works if all the focks have the
    same dimension (e.g., for a single molecule)
    """
    return blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs, vectorized=True)


# ============================================================================
# Loss functions
# ============================================================================


def _loss_eigenvalues_lowdinq_vectorized(
    pred_blocks: TensorMap,
    frames: List[Atoms],
    eigvals: torch.Tensor,
    lowdinq: torch.Tensor,
    orbs: Dict[int, List],
    ao_labels: List[int],
    nelec_dict: Dict[str, float],
    regloss: torch.Tensor,
    weight_eigvals: float = 1.0,
    weight_lowdinq: float = 1.0,
    weight_regloss: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""combined loss on MO energies and Lowdin charges

    Computes a (regularized) combined loss on MO energies and
    Lowdin charges.

        L = w_eigvals * Σ_fr |pred_ε_fr - ε_fr|^2
          + w_lowdinq * Σ_fa |pred_q_fa - q_fa|^2
          + w_regular * L_regularization

    Where f indexes the frame, r indexes the MO, a indexes the atom.
    Every operation is vectorized (fast), so it only works for focks
    that have the same dimension (e.g., for a single molecule).
    """
    # predict fock matrices as torch.Tensors
    pred_focks = _predict_focks_vectorized(pred_blocks, frames=frames, orbs=orbs)

    # MO energies
    pred_eigvals = torch.linalg.eigvalsh(pred_focks)

    # Lowdin charges
    pred_lowdinq, _ = batched_orthogonal_lowdin_population(
        pred_focks,
        nelec_dict,
        ao_labels,
    )

    # MSE loss on energies and charges
    loss_eigvals = torch.mean((eigvals - pred_eigvals) ** 2)
    loss_lowdinq = torch.mean((lowdinq - pred_lowdinq) ** 2)

    # weighted sum of the various loss contributions
    return (
        weight_eigvals * loss_eigvals
        + weight_lowdinq * loss_lowdinq
        + weight_regloss * regloss,
        loss_eigvals,
        loss_lowdinq,
        regloss,
    )


def _loss_eigenvalues_lowdinqbyMO_vectorized(
    pred_blocks: TensorMap,
    frames: List[Atoms],
    eigvals: torch.Tensor,
    lowdinq: torch.Tensor,
    orbs: Dict[int, List],
    ao_labels: List[int],
    nelec_dict: Dict[str, float],
    regloss: torch.Tensor,
    weight_eigvals=1.0,
    weight_lowdinq=1.0,
    weight_lowdinq_tot=1.0,
    weight_regloss=1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""combined loss on MO energies and Lowdin charges MO by MO

    Computes a (regularized) combined loss on MO energies and
    Lowdin charges (per MO).

        L = w_eigvals * Σ_fr |pred_ε_fr - ε_fr|^2
          + w_lowdinq * Σ_fra |pred_q_fra - q_fra|^2
          + w_lowdinq_tot * Σ_fa |pred_q_fa - q_fa|^2
          + w_regular * L_regularization

    Where f indexes the frame, r indexes the MO, a indexes the atom.
    Every operation is vectorized (fast), so it only works for focks
    that have the same dimension (e.g., for a single molecule).
    """
    # predict fock matrices as torch.Tensors
    pred_focks = _predict_focks_vectorized(pred_blocks, frames=frames, orbs=orbs)

    # MO energies
    pred_eigvals = torch.linalg.eigvalsh(pred_focks)

    # Lowdin charges, MO per MO
    # This means they are not an array (n_atoms,), but a matrix
    # (n_mo, n_atoms)
    pred_lowdinq, _ = batched_orthogonal_lowdinbyMO_population(
        pred_focks,
        nelec_dict,
        ao_labels,
    )

    # Lowdin charges per atom (summed over MOs)
    tot_lowdinq = torch.sum(lowdinq, dim=1)
    tot_pred_lowdinq = torch.sum(pred_lowdinq, dim=1)

    # TODO: add _maybe_select_mo_indices that selects only certain MOs
    #       (e.g., to not include core MOs, etc)

    # MSE on energies, lowdin charges per MO, total lowdin charges
    loss_eigvals = torch.mean((eigvals - pred_eigvals) ** 2)
    loss_lowdinq = torch.mean((lowdinq - pred_lowdinq) ** 2)
    loss_lowdinq_tot = torch.mean((tot_lowdinq - tot_pred_lowdinq) ** 2)

    return (
        weight_eigvals * loss_eigvals
        + weight_lowdinq * loss_lowdinq
        + weight_regloss * regloss
        + weight_lowdinq_tot * loss_lowdinq_tot,
        loss_eigvals,
        loss_lowdinq,
        regloss,
        loss_lowdinq_tot,
    )


# ============================================================================
# Train helpers
# ============================================================================


def _accumulate_batch_losses(
    losses_dict, total_loss, eig_loss, low_loss, reg_loss, *other_losses
):
    with torch.no_grad():
        losses_dict["total"].append(total_loss.item())
        losses_dict["eig_loss"].append(eig_loss.item())
        losses_dict["low_loss"].append(low_loss.item())
        losses_dict["reg_loss"].append(reg_loss.item())
        for i, loss in enumerate(other_losses):
            losses_dict[f"other_loss_{i}"].append(loss.item())
        return losses_dict


def _compute_average_losses(losses_dict):
    for lname, lvalue in losses_dict.items():
        losses_dict[lname] = np.mean(lvalue)
    return losses_dict


# ============================================================================
# Models for a Single Molecule
# ============================================================================


class RidgeOnEnergiesAndLowdin(RidgeModel):
    def __init__(
        self,
        coupled_tmap: TensorMap,
        features: TensorMap,
        alpha: float = 1.0,
        dump_dir: str = "",
        bias: bool = False,
    ) -> None:
        """
        Args:
            coupled_tmap: Fock matrix in the coupled basis
                          The features of this matrix (what kind of blocks
                          to predict, ...) are used to set up the list
                          of models, one for each irrep
            features: Features, used to set up the dimensions of the
                      linear models' weight matrices
            alpha: regularization strength
            dump_dir: folder to which the model state is dumped
            bias: whether to use a bias in the linear models. For equivariant
                  learning, use False.
        """
        super().__init__(
            coupled_tmap=coupled_tmap,
            features=features,
            alpha=alpha,
            dump_dir=dump_dir,
            bias=bias,
        )

    def loss_fn(
        self,
        pred_blocks: TensorMap,
        frames: List[Atoms],
        eigvals: torch.Tensor,
        lowdinq: torch.Tensor,
        orbs: Dict[int, List],
        ao_labels: List[int],
        nelec_dict: Dict[str, float],
        weight_eigvals=1.5e6,
        weight_lowdinq=1e6,
        weight_regloss=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _loss_eigenvalues_lowdinq_vectorized(
            pred_blocks=pred_blocks,
            frames=frames,
            eigvals=eigvals,
            lowdinq=lowdinq,
            orbs=orbs,
            ao_labels=ao_labels,
            nelec_dict=nelec_dict,
            regloss=self.regloss_,
            weight_eigvals=weight_eigvals,
            weight_lowdinq=weight_lowdinq,
            weight_regloss=weight_regloss,
        )

    def fit(
        self,
        train_dataset: Dataset,
        epochs: int = 1,
        optim_kwargs: Dict[str, Any] = dict(),
        verbose: int = 10,
        dump: int = 10,
        loss_kwargs: Dict[str, Any] = None,
    ):
        """
        Args:
            train_dataset: training dataset
            epochs: number of epochs
            optim_kwargs: keyword arguments passed to the torch optimizer
            verbose: how many epochs to run before updating history, progress bar
            dump: how many epochs to run before dumping the model state
            loss_kwargs: keyword arguments passed to the loss function
        """
        if loss_kwargs is None:
            loss_kwargs = dict()

        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)

        iterator = tqdm(range(epochs), ncols=120)
        for epoch in iterator:
            losses = defaultdict(list)
            for idx in range(len(train_dataset)):
                x, frames, eigvals, lowdinq = train_dataset[idx]

                optimizer.zero_grad()
                pred = self(x)

                # loss + regularization
                loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=train_dataset.orbs,
                    ao_labels=train_dataset.ao_labels,
                    nelec_dict=train_dataset.nelec_dict,
                    **loss_kwargs,
                )

                loss.backward()
                optimizer.step()

                # accumulate losses in the batch
                losses = _accumulate_batch_losses(losses, loss, *other_losses)

            # average loss in the batch
            with torch.no_grad():
                losses = _compute_average_losses(losses)

                # update progress bar and history
                if epoch % verbose == 0:
                    iterator.set_postfix(**losses)
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self


class RidgeOnEnergiesAndLowdinByMO(RidgeOnEnergiesAndLowdin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_fn(
        self,
        pred_blocks: TensorMap,
        frames: List[Atoms],
        eigvals: torch.Tensor,
        lowdinq: torch.Tensor,
        orbs: Dict[int, List],
        ao_labels: List[int],
        nelec_dict: Dict[str, float],
        weight_eigvals=1.5e6,
        weight_lowdinq=1e6,
        weight_lowdinq_tot=1e6,
        weight_regloss=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _loss_eigenvalues_lowdinqbyMO_vectorized(
            pred_blocks=pred_blocks,
            frames=frames,
            eigvals=eigvals,
            lowdinq=lowdinq,
            orbs=orbs,
            ao_labels=ao_labels,
            nelec_dict=nelec_dict,
            regloss=self.regloss_,
            weight_eigvals=weight_eigvals,
            weight_lowdinq=weight_lowdinq,
            weight_lowdinq_tot=weight_lowdinq_tot,
            weight_regloss=weight_regloss,
        )


# ============================================================================
# Models for Multiple Molecles
# ============================================================================


class RidgeOnEnergiesAndLowdinMultipleMolecules(RidgeModel):
    def __init__(
        self,
        coupled_tmap: TensorMap,
        features: TensorMap,
        alpha: float = 1.0,
        dump_dir: str = "",
        bias: bool = False,
    ) -> None:
        """
        Args:
            coupled_tmap: Fock matrix in the coupled basis
                          The features of this matrix (what kind of blocks
                          to predict, ...) are used to set up the list
                          of models, one for each irrep
            features: Features, used to set up the dimensions of the
                      linear models' weight matrices
            alpha: regularization strength
            dump_dir: folder to which the model state is dumped
            bias: whether to use a bias in the linear models. For equivariant
                  learning, use False.
        """
        super().__init__(
            coupled_tmap=coupled_tmap,
            features=features,
            alpha=alpha,
            dump_dir=dump_dir,
            bias=bias,
        )

    def loss_fn(
        self,
        pred_blocks: TensorMap,
        frames: List[Atoms],
        eigvals: torch.Tensor,
        lowdinq: torch.Tensor,
        orbs: Dict[int, List],
        ao_labels: List[int],
        nelec_dict: Dict[str, float],
        weight_eigvals=1.5e6,
        weight_lowdinq=1e6,
        weight_regloss=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _loss_eigenvalues_lowdinq_vectorized(
            pred_blocks=pred_blocks,
            frames=frames,
            eigvals=eigvals,
            lowdinq=lowdinq,
            orbs=orbs,
            ao_labels=ao_labels,
            nelec_dict=nelec_dict,
            regloss=self.regloss_,
            weight_eigvals=weight_eigvals,
            weight_lowdinq=weight_lowdinq,
            weight_regloss=weight_regloss,
        )

    def fit(
        self,
        train_datasets: List[Dataset],
        epochs: int = 1000,
        optim_kwargs: Dict[str, Any] = dict(),
        verbose: int = 10,
        dump: int = 10,
        loss_kwargs: Dict[str, Any] = None,
    ):
        """
        Args:
            train_dataset: training dataset
            epochs: number of epochs
            optim_kwargs: keyword arguments passed to the torch optimizer
            verbose: how many epochs to run before updating history, progress bar
            dump: how many epochs to run before dumping the model state
            loss_kwargs: keyword arguments passed to the loss function
        """
        if loss_kwargs is None:
            loss_kwargs = dict()

        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)

        iterator = tqdm(range(epochs), ncols=120)
        for epoch in iterator:
            losses = defaultdict(list)
            for train_dataset in train_datasets:
                for idx in range(len(train_dataset)):
                    x, frames, eigvals, lowdinq = train_dataset[idx]

                    optimizer.zero_grad()
                    pred = self(x)

                    loss, *other_losses = self.loss_fn(
                        pred_blocks=pred,
                        frames=frames,
                        eigvals=eigvals,
                        lowdinq=lowdinq,
                        orbs=train_dataset.orbs,
                        ao_labels=train_dataset.ao_labels,
                        nelec_dict=train_dataset.nelec_dict,
                        **loss_kwargs,
                    )

                    loss.backward()
                    optimizer.step()

                    losses = _accumulate_batch_losses(losses, loss, *other_losses)

            # average loss over batches and molecules
            with torch.no_grad():
                losses = _compute_average_losses(losses)

                # update progress bar and history
                if epoch % verbose == 0:
                    iterator.set_postfix(**losses)
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self


class RidgeOnEnergiesAndLowdinMultipleMoleculesByMO(
    RidgeOnEnergiesAndLowdinMultipleMolecules
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_fn(
        self,
        pred_blocks: TensorMap,
        frames: List[Atoms],
        eigvals: List[torch.Tensor],
        lowdinq: List[torch.Tensor],
        orbs: Dict[int, List[Tuple[int, int, int]]],
        ao_labels: List[List[Any]],
        nelec_dict: Dict[str, float],
        weight_eigvals=1.5e6,
        weight_lowdinq=1e6,
        weight_lowdinq_tot=1e6,
        weight_regloss=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _loss_eigenvalues_lowdinqbyMO_vectorized(
            pred_blocks=pred_blocks,
            frames=frames,
            eigvals=eigvals,
            lowdinq=lowdinq,
            orbs=orbs,
            ao_labels=ao_labels,
            nelec_dict=nelec_dict,
            regloss=self.regloss_,
            weight_eigvals=weight_eigvals,
            weight_lowdinq=weight_lowdinq,
            weight_lowdinq_tot=weight_lowdinq_tot,
            weight_regloss=weight_regloss,
        )


# class RidgeOnEnergiesAndLowdinByMO_2(RidgeOnEnergiesAndLowdin):
#     def __init__(self, *args, **kwargs):
#         self.skip_n_mo = kwargs.pop("skip_n_mo")
#         super().__init__(*args, **kwargs)
#
#     def loss_fn(
#         self,
#         pred_blocks: TensorMap,
#         frames: List[Atoms],
#         eigvals: torch.Tensor,
#         lowdinq: torch.Tensor,
#         orbs: Dict[int, List],
#         ao_labels: List[int],
#         nelec_dict: Dict[str, float],
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         pred_focks = _predict_focks_vectorized(pred_blocks, frames=frames, orbs=orbs)
#
#         pred_eigvals = torch.linalg.eigvalsh(pred_focks)
#
#         # lowdin MO by MO
#         pred_lowdinq, _ = batched_orthogonal_lowdinbyMO_population(
#             pred_focks,
#             nelec_dict,
#             ao_labels,
#         )
#
#         loss_a = torch.mean((eigvals - pred_eigvals) ** 2)
#         loss_b = torch.mean(
#             (lowdinq[:, self.skip_n_mo :] - pred_lowdinq[:, self.skip_n_mo :]) ** 2
#         )
#
#         return (
#             1.5e6 * loss_a + 1e6 * loss_b + self.regloss_,
#             loss_a,
#             loss_b,
#             self.regloss_,
#         )
