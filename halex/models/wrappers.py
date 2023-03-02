from __future__ import annotations

from collections import defaultdict
from .tmap_models import RidgeModel
from ..popan import batched_orthogonal_lowdin_population
from ..hamiltonian import blocks_to_dense, decouple_blocks

import numpy as np
import torch

from tqdm import tqdm


class RidgeOnEnergiesAndLowdin(RidgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_fn(
        self,
        focks,
        pred_blocks,
        frames,
        eigvals,
        lowdinq,
        orbs,
        ao_labels,
        nelec_dict,
    ):
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
        train_dataset,
        epochs=1000,
        batch_size=64,
        optim_kwargs=dict(),
        verbose=10,
        dump=10,
    ):
        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)

        batch_indices = train_dataset.get_indices(batch_size=batch_size)

        iterator = tqdm(range(epochs), ncols=120)
        for epoch in iterator:
            losses = defaultdict(list)
            for idx in batch_indices:
                x, y, frames, eigvals, lowdinq = train_dataset[idx]

                optimizer.zero_grad()
                pred = self(x)

                # loss + regularization
                loss, eig_loss, low_loss = self.loss_fn(
                    focks=y,
                    pred_blocks=pred,
                    frames=frames,
                    eigvals=eigvals,
                    lowdinq=lowdinq,
                    orbs=train_dataset.orbs,
                    ao_labels=train_dataset.ao_labels,
                    nelec_dict=train_dataset.nelec_dict,
                )
                for model in self.models:
                    loss += model.regularization_loss(y=y, pred=pred)

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
