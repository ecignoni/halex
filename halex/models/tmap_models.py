from __future__ import annotations
from typing import Dict

import os
from collections import defaultdict

import numpy as np
import torch

from sklearn.linear_model import Ridge

from equistore import Labels, TensorBlock, TensorMap

from .block_models import RidgeBlockModel


class RidgeModel(torch.nn.Module):
    def __init__(
        self,
        coupled_tmap: TensorMap,
        features: TensorMap,
        core_features: TensorMap = None,
        alpha: float = 1.0,
        dump_dir: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.dump_dir = dump_dir
        self.bias = bias
        self._setup_block_models(
            coupled_tmap=coupled_tmap, features=features, core_features=core_features
        )
        self.reset_history()

    def coupled_fock_key_is_core(self, key: Labels):
        """true if a key correspond to core-shell orbitals

        This is a very raw implementation and only works
        for second row atoms, as it considers only the 1s
        AOs as forming the core MOs (except for hydrogen)
        """
        block_type, ai, ni, li, aj, nj, lj, L = key
        c0 = block_type == 0  # same atoms
        c1 = ai != 1  # ignore hydrogens
        c2 = ni == 1  # inner shell
        c3 = li == 0  # s type orbital
        c4 = aj != 1
        c5 = nj == 1
        c6 = lj == 0
        c7 = L == 0  # obvious but ok
        return c0 and c1 and c2 and c3 and c4 and c5 and c6 and c7

    def maybe_select_core_features(
        self, key: Labels, features: TensorMap, core_features: TensorMap = None
    ) -> TensorMap:
        "returns core_features if the key correspond to a core element of the fock"
        if core_features is None:
            return features
        return core_features if self.coupled_fock_key_is_core(key) else features

    def get_feature_block(
        self, features: TensorMap, key: Labels, core_features: TensorMap = None
    ) -> TensorBlock:
        feats = self.maybe_select_core_features(
            key=key, features=features, core_features=core_features
        )
        block_type, ai, ni, li, aj, nj, lj, L = key
        inversion_sigma = (-1) ** (li + lj + L)
        block = feats.block(
            block_type=block_type,
            spherical_harmonics_l=L,
            inversion_sigma=inversion_sigma,
            species_center=ai,
            species_neighbor=aj,
        )
        return block

    def _setup_block_models(
        self,
        coupled_tmap: TensorMap,
        features: TensorMap,
        core_features: TensorMap = None,
    ) -> None:
        components = []
        properties = Labels(["values"], np.asarray([[0]], dtype=np.int32))
        models = []
        for key, block in coupled_tmap:
            nsamples, ncomps, nprops = self.get_feature_block(
                features=features, key=key, core_features=core_features
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

    def reset_history(self) -> None:
        self.history = defaultdict(list)

    def update_history(self, losses: Dict[str, float]) -> None:
        for loss_name, loss_value in losses.items():
            self.history[loss_name].append(loss_value)

    def dump_state(self) -> None:
        history_path = os.path.join(self.dump_dir, "history.npz")
        state_dict_path = os.path.join(self.dump_dir, "model_state_dict.pth")
        np.savez(history_path, **self.history)
        torch.save(self.state_dict(), state_dict_path)

    def forward(
        self, features: TensorMap, core_features: TensorMap = None
    ) -> TensorMap:
        self.regloss_ = 0.0
        pred_blocks = []
        for key, components, model in zip(
            self.predict_keys, self.predict_components, self.models
        ):
            L = key["L"]
            feat_block = self.get_feature_block(
                features=features, key=key, core_features=core_features
            )
            x = feat_block.values
            nsamples, ncomps, nprops = x.shape
            x = x.reshape(nsamples * ncomps, nprops)
            pred = model(x)
            self.regloss_ += model.regularization_loss(pred)

            pred_block = TensorBlock(
                values=pred.reshape((-1, 2 * L + 1, 1)),
                samples=feat_block.samples,
                components=components,
                properties=self.predict_properties,
            )
            pred_blocks.append(pred_block)
        pred_tmap = TensorMap(self.predict_keys, pred_blocks)
        return pred_tmap

    def fit_ridge_analytical(
        self, features: TensorMap, targets: TensorMap, core_features: TensorMap = None
    ) -> None:
        for key, model in zip(self.predict_keys, self.models):
            feat_block = self.get_feature_block(
                features=features, key=key, core_features=core_features
            )
            targ_block = targets[key]

            x = np.array(feat_block.values.reshape(-1, feat_block.values.shape[2]))
            y = np.array(targ_block.values.reshape(-1, 1))

            ridge = Ridge(alpha=self.alpha, fit_intercept=self.bias).fit(x, y)

            model.layer.weight = torch.nn.Parameter(
                torch.from_numpy(ridge.coef_.copy().astype(np.float64))
            )
            if self.bias:
                model.layer.bias = torch.nn.Parameter(
                    torch.from_numpy(ridge.intercept_.copy().astype(np.float64))
                )
