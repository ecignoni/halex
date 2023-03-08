from __future__ import annotations
from typing import Dict, List, Any, Tuple

from collections import defaultdict
from .tmap_models import RidgeModel
from ..popan import batched_orthogonal_lowdin_population, orthogonal_lowdin_population
from ..hamiltonian import blocks_to_dense, decouple_blocks

import numpy as np
import torch

from tqdm import tqdm

from equistore import TensorMap

Dataset = Any
Atoms = Any


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
        pred_focks = torch.stack(
            blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
        )
        pred_eigvals = torch.linalg.eigvalsh(pred_focks)
        pred_lowdinq, _ = batched_orthogonal_lowdin_population(
            pred_focks,
            nelec_dict,
            ao_labels,
        )
        loss_a = torch.mean((eigvals - pred_eigvals) ** 2)
        loss_b = torch.mean((lowdinq - pred_lowdinq) ** 2)
        return 1.5e6 * loss_a + 1e6 * loss_b, loss_a, loss_b

    def fit(
        self,
        train_dataset: Dataset,
        epochs: int = 1000,
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
                loss, eig_loss, low_loss = self.loss_fn(
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
                    losses["eig_loss"].append(loss.item())
                    losses["low_loss"].append(loss.item())

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
                    )
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self


class RidgeOnEnergiesAndLowdinMultipleMolecules(RidgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_fn(
        pred_blocks: TensorMap,
        frames: List[Atoms],
        eigvals: List[torch.Tensor],
        lowdinq: List[torch.Tensor],
        orbs: Dict[int, List[Tuple[int, int, int]]],
        ao_labels: List[List[Any]],
        nelec_dict: Dict[str, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Here the dimension of eigenvalues and lowdin charges is, in general, different,
        # as molecule can have different atoms. We are then bound to use lists and loops
        # pure Python
        pred_focks = blocks_to_dense(decouple_blocks(pred_blocks), frames, orbs)
        pred_eigvals = [torch.linalg.eigvalsh(f) for f in pred_focks]
        pred_lowdinq = [
            orthogonal_lowdin_population(f, nelec_dict, ao)[0]
            for f, ao in zip(pred_focks, ao_labels)
        ]

        loss_a = 0.0
        for pred_eigval, eigval in zip(pred_eigvals, eigvals):
            # mean over atoms
            loss_a += torch.mean((pred_eigval - eigval) ** 2)
        # mean over samples
        loss_a = torch.mean(loss_a)

        loss_b = 0.0
        for pred_lowq, lowq in zip(pred_lowdinq, lowdinq):
            # mean over atoms
            loss_b += torch.mean((pred_lowq - lowq) ** 2)
        # mean over samples
        loss_b = torch.mean(loss_b)

        return 1.5e6 * loss_a + 1e6 * loss_b, loss_a, loss_b

    def fit(
        self,
        train_dataset: Dataset,
        epochs: int = 1000,
        optim_kwargs: Dict[str, Any] = dict(),
        verbose: int = 10,
        dump: int = 10,
    ) -> Self:  # noqa
        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)

        iterator = tqdm(range(epochs), ncols=120)
        for epoch in iterator:
            losses = defaultdict(list)
            for idx in range(len(train_dataset)):
                x, frames, eigvals, lowdinq, ao_labels = train_dataset[idx]

                optimizer.zero_grad()
                pred = self(x)

                # loss + regularization
                loss, eig_loss, low_loss = self.loss_fn(
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=train_dataset.orbs,
                    ao_labels=ao_labels,
                    nelec_dict=train_dataset.nelec_dict,
                )
                # for model in self.models:
                #     loss += model.regularization_loss(pred=pred)

                loss.backward()
                optimizer.step()

                # accumulate losses in the batch
                with torch.no_grad():
                    losses["total"].append(loss.item())
                    losses["eig_loss"].append(loss.item())
                    losses["low_loss"].append(loss.item())

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
                    )
                    self.update_history(losses)

                if epoch % dump == 0:
                    self.dump_state()

        return self

    def fit_ridge_analytical(self, features: TensorMap, targets: TensorMap) -> None:
        for key, model in zip(self.predict_keys, self.models):
            feat_block = self.get_feature_block(features, key)
            targ_block = targets[key]

            if feat_block.values.shape[0] != targ_block.values.shape[0]:
                print("features", feat_block.samples)
                print("features", targ_block.samples)
                print(key)
                break

            # x = np.array(feat_block.values.reshape(-1, feat_block.values.shape[2]))
            # y = np.array(targ_block.values.reshape(-1, 1))

            # ridge = Ridge(alpha=self.alpha, fit_intercept=self.bias).fit(x, y)

            # model.layer.weight = torch.nn.Parameter(
            #     torch.from_numpy(ridge.coef_.copy().astype(np.float64))
            # )
            # if self.bias:
            #     model.layer.bias = torch.nn.Parameter(
            #         torch.from_numpy(ridge.intercept_.copy().astype(np.float64))
            #     )
