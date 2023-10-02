from __future__ import annotations
from typing import Dict, List, Any, Tuple

from collections import defaultdict
from .tmap_models import RidgeModel
from ..popan import batched_orthogonal_lowdin_population

from ..hamiltonian import blocks_to_dense, decouple_blocks

import numpy as np
import torch

from tqdm import tqdm

from metatensor import TensorMap

Atoms = Any
Dataset = Any
Optimizer = Any


def _predict_focks_vectorized(pred_blocks, frames, orbs):
    """Converts the block tensormap back to dense fock matrices

    This is a vectorized (faster evaluation, faster backpropagation)
    version of the prediction. Only works if all the focks have the
    same dimension (e.g., for a single molecule)
    """
    return blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs, vectorized=True)


# ============================================================================
# Loss functions
# ============================================================================
def _loss_focks_vectorized(
    pred_blocks: TensorMap,
    frames: List[Atoms],
    focks: torch.Tensor,
    orbs: Dict[int, List],
    regloss: torch.Tensor,
    weight_focks: float = 1.0,
    weight_regloss: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Loss on the full fock matrices

    Computes a (regularized) loss on Fock matrices.

        L = w_focks * Σ_fr |pred_fock_fr - fock_fr|^2
        + w_regular * L_regularization

    Where f indexes the frame and r indexes the MO.
    Every operation is vectorized (fast), so it only works for focks
    that have the same dimension (e.g., for a single molecule).
    """

    pred_focks = _predict_focks_vectorized(pred_blocks, frames=frames, orbs=orbs)
    loss_focks = torch.mean((focks - pred_focks) ** 2)

    return (weight_focks * loss_focks + weight_regloss * regloss, regloss)


def _loss_eigenvalues_vectorized(
    pred_blocks: TensorMap,
    frames: List[Atoms],
    eigvals: torch.Tensor,
    orbs: Dict[int, List],
    ao_labels: List[int],
    nelec_dict: Dict[str, float],
    regloss: torch.Tensor,
    weight_eigvals: float = 1.0,
    weight_regloss: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""loss on MO energies.

    Computes a (regularized) loss on MO energies.

        L = w_eigvals * Σ_fr |pred_ε_fr - ε_fr|^2
          + w_regular * L_regularization

    Where f indexes the frame, r indexes the MO.
    Every operation is vectorized (fast), so it only works for focks
    that have the same dimension (e.g., for a single molecule).
    """
    # predict fock matrices as torch.Tensors
    pred_focks = _predict_focks_vectorized(pred_blocks, frames=frames, orbs=orbs)

    # MO energies
    pred_eigvals = torch.linalg.eigvalsh(pred_focks)

    # MSE loss on energies
    loss_eigvals = torch.mean((eigvals - pred_eigvals) ** 2)

    # weighted sum of the various loss contributions
    return (
        weight_eigvals * loss_eigvals + weight_regloss * regloss,
        loss_eigvals,
        regloss,
    )


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
    **kwargs,
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


# ============================================================================
# Train helpers
# ============================================================================


def _get_batches_from_dataset(dataset, idx):
    try:
        x, x_core, focks, frames, eigvals, lowdinq = dataset[idx]
    except ValueError:
        x, focks, frames, eigvals, lowdinq = dataset[idx]
        x_core = None
    return x, x_core, focks, frames, eigvals, lowdinq


def _accumulate_batch_losses(
    losses_dict,
    total_loss,
    eig_loss,
    low_loss,
    reg_loss,
    *other_losses,
    prefix="",
):
    with torch.no_grad():
        losses_dict[f"{prefix}total"].append(total_loss.item())
        losses_dict[f"{prefix}eig_loss"].append(eig_loss.item())
        losses_dict[f"{prefix}low_loss"].append(low_loss.item())
        losses_dict[f"{prefix}reg_loss"].append(reg_loss.item())
        for i, loss in enumerate(other_losses):
            losses_dict[f"{prefix}other_loss_{i}"].append(loss.item())
        return losses_dict


def _compute_average_losses(losses_dict):
    for lname, lvalue in losses_dict.items():
        losses_dict[lname] = np.mean(lvalue)
    return losses_dict


# ============================================================================
# Models for Multiple Molecles
# ============================================================================


class RidgeOnFocksMultipleMolecules(RidgeModel):
    """
    A model that predicts the Fock matrix in the coupled basis
    and then converts it to uncoupled dense matrix and computes
    the loss on it and backpropagates it via gradient descent
    to optimize the model.
    """

    def __init__(
        self,
        coupled_tmap: TensorMap,
        features: TensorMap,
        core_features: TensorMap = None,
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
            core_features=core_features,
            alpha=alpha,
            dump_dir=dump_dir,
            bias=bias,
        )

    def loss_fn(
        self,
        pred_blocks: TensorMap,
        frames: List[Atoms],
        focks: torch.Tensor,
        orbs: Dict[int, List],
        weight_focks=1.0,
        weight_regloss=1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _loss_focks_vectorized(
            pred_blocks=pred_blocks,
            frames=frames,
            focks=focks,
            orbs=orbs,
            regloss=self.regloss_,
            weight_focks=weight_focks,
            weight_regloss=weight_regloss,
            **kwargs,
        )

    def _train_step(
        self,
        train_datasets: List[Dataset],
        optimizer: Optimizer,
        loss_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        losses = defaultdict(list)

        for train_dataset in train_datasets:
            for idx in range(len(train_dataset)):
                x, x_core, focks, frames, _, _ = _get_batches_from_dataset(
                    train_dataset, idx
                )

                optimizer.zero_grad()
                pred = self(features=x, core_features=x_core)

                loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    focks=focks,
                    orbs=train_dataset.orbs,
                    **loss_kwargs,
                )

                loss.backward()
                optimizer.step()

                losses = _accumulate_batch_losses(
                    losses, loss, *other_losses, prefix="train_"
                )

        with torch.no_grad():
            losses = _compute_average_losses(losses)

        return losses

    @torch.no_grad()
    def _validation_step(
        self, valid_datasets: List[Dataset], loss_kwargs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        losses = defaultdict(list)

        for valid_dataset in valid_datasets:
            for idx in range(len(valid_dataset)):
                x, x_core, focks, frames, _, _ = _get_batches_from_dataset(
                    valid_dataset, idx
                )

                pred = self(features=x, core_features=x_core)

                loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    focks=focks,
                    orbs=valid_dataset.orbs,
                    **loss_kwargs,
                )

                losses = _accumulate_batch_losses(
                    losses, loss, *other_losses, prefix="valid_"
                )

        losses = _compute_average_losses(losses)
        return losses

    def fit(
        self,
        train_datasets: List[Dataset],
        valid_datasets: List[Dataset] = None,
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
            losses = self._train_step(train_datasets, optimizer, loss_kwargs)

            if valid_datasets is not None:
                valid_losses = self._validation_step(valid_datasets, loss_kwargs)
                losses |= valid_losses

            # average loss over batches and molecules
            with torch.no_grad():
                # update progress bar and history
                if epoch % verbose == 0:
                    iterator.set_postfix(
                        {k: v for k, v in losses.items() if k.endswith("total")}
                    )
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self


class RidgeOnEnergiesMultipleMolecules(RidgeModel):
    """
    A model that predicts the Fock matrix in the coupled basis
    and then converts it to uncoupled dense matrix, from which the
    eigenvalues are computed and the loss on the eigenvalues are then
    backpropagated via gradient descent to optimize the model.
    """

    def __init__(
        self,
        coupled_tmap: TensorMap,
        features: TensorMap,
        core_features: TensorMap = None,
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
            core_features=core_features,
            alpha=alpha,
            dump_dir=dump_dir,
            bias=bias,
        )

    def loss_fn(
        self,
        pred_blocks: TensorMap,
        frames: List[Atoms],
        eigvals: torch.Tensor,
        orbs: Dict[int, List],
        ao_labels: List[int],
        nelec_dict: Dict[str, float],
        weight_eigvals=1.5e6,
        weight_regloss=1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _loss_eigenvalues_vectorized(
            pred_blocks=pred_blocks,
            frames=frames,
            eigvals=eigvals,
            orbs=orbs,
            ao_labels=ao_labels,
            nelec_dict=nelec_dict,
            regloss=self.regloss_,
            weight_eigvals=weight_eigvals,
            weight_regloss=weight_regloss,
            **kwargs,
        )

    def _train_step(
        self,
        train_datasets: List[Dataset],
        optimizer: Optimizer,
        loss_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        losses = defaultdict(list)

        for train_dataset in train_datasets:
            for idx in range(len(train_dataset)):
                x, x_core, _, frames, eigvals, lowdinq = _get_batches_from_dataset(
                    train_dataset, idx
                )

                optimizer.zero_grad()
                pred = self(features=x, core_features=x_core)

                loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=train_dataset.orbs,
                    ao_labels=train_dataset.ao_labels,
                    nelec_dict=train_dataset.nelec_dict,
                    mo_indices=train_dataset.lowdin_mo_indices,
                    **loss_kwargs,
                )

                loss.backward()
                optimizer.step()

                losses = _accumulate_batch_losses(
                    losses, loss, *other_losses, prefix="train_"
                )

        with torch.no_grad():
            losses = _compute_average_losses(losses)

        return losses

    @torch.no_grad()
    def _validation_step(
        self, valid_datasets: List[Dataset], loss_kwargs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        losses = defaultdict(list)

        for valid_dataset in valid_datasets:
            for idx in range(len(valid_dataset)):
                x, x_core, _, frames, eigvals, lowdinq = _get_batches_from_dataset(
                    valid_dataset, idx
                )

                pred = self(features=x, core_features=x_core)

                loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=valid_dataset.orbs,
                    ao_labels=valid_dataset.ao_labels,
                    nelec_dict=valid_dataset.nelec_dict,
                    mo_indices=valid_dataset.lowdin_mo_indices,
                    **loss_kwargs,
                )

                losses = _accumulate_batch_losses(
                    losses, loss, *other_losses, prefix="valid_"
                )

        losses = _compute_average_losses(losses)
        return losses

    def fit(
        self,
        train_datasets: List[Dataset],
        valid_datasets: List[Dataset] = None,
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
            losses = self._train_step(train_datasets, optimizer, loss_kwargs)

            if valid_datasets is not None:
                valid_losses = self._validation_step(valid_datasets, loss_kwargs)
                losses |= valid_losses

            # average loss over batches and molecules
            with torch.no_grad():
                # update progress bar and history
                if epoch % verbose == 0:
                    iterator.set_postfix(
                        {k: v for k, v in losses.items() if k.endswith("total")}
                    )
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self


class RidgeOnEnergiesAndLowdinMultipleMolecules(RidgeModel):
    """
    A model that predicts the Fock matrix in the coupled basis
    and then converts it to uncoupled dense matrix, from which the
    eigenvalues and Lowdin charges are computed and the combined loss
    on them is then backpropagated.
    """

    def __init__(
        self,
        coupled_tmap: TensorMap,
        features: TensorMap,
        core_features: TensorMap = None,
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
            core_features=core_features,
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
        **kwargs,
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
            **kwargs,
        )

    def _train_step(
        self,
        train_datasets: List[Dataset],
        optimizer: Optimizer,
        loss_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        losses = defaultdict(list)

        for train_dataset in train_datasets:
            for idx in range(len(train_dataset)):
                x, x_core, _, frames, eigvals, lowdinq = _get_batches_from_dataset(
                    train_dataset, idx
                )

                optimizer.zero_grad()
                pred = self(features=x, core_features=x_core)

                loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=train_dataset.orbs,
                    ao_labels=train_dataset.ao_labels,
                    nelec_dict=train_dataset.nelec_dict,
                    mo_indices=train_dataset.lowdin_mo_indices,
                    **loss_kwargs,
                )

                loss.backward()
                optimizer.step()

                losses = _accumulate_batch_losses(
                    losses, loss, *other_losses, prefix="train_"
                )

        with torch.no_grad():
            losses = _compute_average_losses(losses)

        return losses

    @torch.no_grad()
    def _validation_step(
        self, valid_datasets: List[Dataset], loss_kwargs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        losses = defaultdict(list)

        for valid_dataset in valid_datasets:
            for idx in range(len(valid_dataset)):
                x, x_core, _, frames, eigvals, lowdinq = _get_batches_from_dataset(
                    valid_dataset, idx
                )

                pred = self(features=x, core_features=x_core)

                loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=valid_dataset.orbs,
                    ao_labels=valid_dataset.ao_labels,
                    nelec_dict=valid_dataset.nelec_dict,
                    mo_indices=valid_dataset.lowdin_mo_indices,
                    **loss_kwargs,
                )

                losses = _accumulate_batch_losses(
                    losses, loss, *other_losses, prefix="valid_"
                )

        losses = _compute_average_losses(losses)
        return losses

    def fit(
        self,
        train_datasets: List[Dataset],
        valid_datasets: List[Dataset] = None,
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
            losses = self._train_step(train_datasets, optimizer, loss_kwargs)

            if valid_datasets is not None:
                valid_losses = self._validation_step(valid_datasets, loss_kwargs)
                losses |= valid_losses

            # average loss over batches and molecules
            with torch.no_grad():
                # update progress bar and history
                if epoch % verbose == 0:
                    iterator.set_postfix(
                        {k: v for k, v in losses.items() if k.endswith("total")}
                    )
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self
