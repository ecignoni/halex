from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch

from equistore import Labels, TensorBlock, TensorMap


class RidgeBlockModel(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, alpha: int = 1.0
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.alpha = alpha
        self.layer = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def regularization_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # normalize by the number of samples
        return (
            self.alpha
            * torch.squeeze(self.layer.weight.T @ self.layer.weight)
            / pred.shape[0]
        )


class RidgeModel(torch.nn.Module):
    def __init__(self, coupled_tmap, features, alpha=1.0, dump_dir="", bias=False):
        super().__init__()
        self.alpha = alpha
        self.dump_dir = dump_dir
        self.bias = bias
        self._setup_block_models(coupled_tmap, features)
        self.history = self.reset_history()

    def get_feature_block(self, features, idx):
        block_type, ai, ni, li, aj, nj, lj, L = idx
        inversion_sigma = (-1) ** (li + lj + L)
        block = features.block(
            block_type=block_type,
            spherical_harmonics_l=L,
            inversion_sigma=inversion_sigma,
            species_center=ai,
            species_neighbor=aj,
        )
        return block

    def _setup_block_models(self, coupled_tmap, features):
        components = []
        properties = Labels(["values"], np.asarray([[0]], dtype=np.int32))
        models = []
        for key, block in coupled_tmap:
            nsamples, ncomps, nprops = self.get_feature_block(
                features, key
            ).values.shape
            components.append(block.components.copy())
            module = RidgeBlockModel(
                in_features=nprops, out_features=1, bias=self.bias, alpha=self.alpha
            )
            models.append(module)
        self.predict_keys = coupled_tmap.keys.copy()
        self.predict_components = components
        self.predict_properties = properties
        self.models = torch.nn.ModuleList(models)

    def reset_history(self):
        self.history = defaultdict(list)

    def update_history(self, losses):
        for loss_name, loss_value in losses.items():
            self.history[loss_name].append(loss_value)

    def dump_state(self):
        history_path = os.path.join(self.dump_dir, "history.npz")
        state_dict_path = os.path.join(self.dump_dir, "model_state_dict.pth")
        np.savez(history_path, **self.history)
        torch.save(self.state_dict(), state_dict_path)

    def forward(self, features):
        pred_blocks = []
        for key, components, model in zip(
            self.predict_keys, self.predict_components, self.models
        ):
            L = key["L"]
            feat_block = self.get_feature_block(features, key)
            x = feat_block.values
            nsamples, ncomps, nprops = x.shape
            x = x.reshape(nsamples * ncomps, nprops)
            pred = model(x)

            pred_block = TensorBlock(
                values=pred.reshape((-1, 2 * L + 1, 1)),
                samples=feat_block.samples,
                components=components,
                properties=self.predict_properties,
            )
            pred_blocks.append(pred_block)
        pred_tmap = TensorMap(self.predict_keys, pred_blocks)
        return pred_tmap
