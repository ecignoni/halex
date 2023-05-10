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
        #     loss_lowdinq_tot,
    )


class RidgeOnEnergiesAndLowdin(RidgeModel):
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
            weight_eigvals=1.5e6,
            weight_lowdinq=1e6,
            weight_regloss=1.0,
        )

    def fit(
        self,
        train_dataset: Dataset,
        epochs: int = 1,
        optim_kwargs: Dict[str, Any] = dict(),
        verbose: int = 10,
        dump: int = 10,
    ) -> Self:  # noqa
        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)

        iterator = tqdm(range(epochs), ncols=120)
        for epoch in iterator:
            losses = defaultdict(list)
            for idx in range(len(train_dataset)):
                x, frames, eigvals, lowdinq = train_dataset[idx]

                optimizer.zero_grad()
                pred = self(x)

                # loss + regularization
                loss, eig_loss, low_loss, reg_loss, *other_losses = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=train_dataset.orbs,
                    ao_labels=train_dataset.ao_labels,
                    nelec_dict=train_dataset.nelec_dict,
                )
                # for model in self.models:
                #     loss += model.regularization_loss(pred=pred)

                loss.backward()
                optimizer.step()

                # accumulate losses in the batch
                with torch.no_grad():
                    losses["total"].append(loss.item())
                    losses["eig_loss"].append(eig_loss.item())
                    losses["low_loss"].append(low_loss.item())
                    losses["reg_loss"].append(reg_loss.item())
                    for ii, oloss in enumerate(other_losses):
                        losses[f"other_loss_{ii}"].append(oloss.item())

            # average loss in the batch
            with torch.no_grad():
                for key, value in losses.items():
                    losses[key] = np.mean(value)

                # update progress bar and history
                if epoch % verbose == 0:
                    iterator.set_postfix(
                        loss=losses["total"],
                        eig_loss=losses["eig_loss"],
                        low_loss=losses["low_loss"],
                        reg_loss=losses["reg_loss"],
                    )
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self


class RidgeOnEnergiesAndLowdinMultipleMolecules(RidgeModel):
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
            weight_eigvals=1.5e6,
            weight_lowdinq=1e6,
            weight_regloss=1.0,
        )

    def fit(
        self,
        train_datasets: List[Dataset],
        epochs: int = 1000,
        optim_kwargs: Dict[str, Any] = dict(),
        verbose: int = 10,
        dump: int = 10,
    ) -> Self:  # noqa
        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)

        iterator = tqdm(range(epochs), ncols=120)
        for epoch in iterator:
            losses = defaultdict(list)
            for train_dataset in train_datasets:
                for idx in range(len(train_dataset)):
                    x, frames, eigvals, lowdinq = train_dataset[idx]

                    optimizer.zero_grad()
                    pred = self(x)

                    loss, eig_loss, low_loss, reg_loss = self.loss_fn(
                        pred_blocks=pred,
                        frames=frames,
                        eigvals=eigvals,
                        lowdinq=lowdinq,
                        orbs=train_dataset.orbs,
                        ao_labels=train_dataset.ao_labels,
                        nelec_dict=train_dataset.nelec_dict,
                    )

                    loss.backward()
                    optimizer.step()

                    # accumulate losses in the batch
                    with torch.no_grad():
                        losses["total"].append(loss.item())
                        losses["eig_loss"].append(eig_loss.item())
                        losses["low_loss"].append(low_loss.item())
                        losses["reg_loss"].append(reg_loss.item())

            # average loss over batches and molecules
            with torch.no_grad():
                for key, value in losses.items():
                    losses[key] = np.mean(value)

                # update progress bar and history
                if epoch % verbose == 0:
                    iterator.set_postfix(
                        loss=losses["total"],
                        eig_loss=losses["eig_loss"],
                        low_loss=losses["low_loss"],
                        reg_loss=losses["reg_loss"],
                    )
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
            weight_eigvals=1.5e6,
            weight_lowdinq=1e6,
            weight_lowdinq_tot=1e6,
            weight_regloss=1.0,
        )


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
            weight_eigvals=1.5e6,
            weight_lowdinq=1e6,
            weight_lowdinq_tot=1e6,
            weight_regloss=1.0,
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
