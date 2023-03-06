# import warnings
import numpy as np
import torch
from equistore import TensorBlock, TensorMap


class EquivariantStandardScaler:
    def __init__(self, with_mean=False, with_std=True, components_to_sample=True):
        # warnings.warn("EquivariantStandardScaler is not well tested: beware.")
        self.with_mean = with_mean
        self.with_std = with_std
        self.components_to_sample = components_to_sample

    def fit(self, tmap):
        self.keys_ = []
        self.mean_ = []
        self.scale_ = []
        self.var_ = []

        for key, block in tmap:
            val = block.values
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).type(torch.float64)
            val = val.clone()

            if self.components_to_sample:
                sdim, cdim, pdim = val.shape
                val = val.reshape(sdim * cdim, pdim)
            mean = torch.mean(val, dim=0, keepdim=True) if self.with_mean else 0.0
            var = (
                torch.var(val, dim=0, unbiased=True, keepdim=True)
                if self.with_std
                else 1.0
            )
            scale = torch.sqrt(var)

            self.keys_.append(key)
            self.mean_.append(mean)
            self.var_.append(var)
            self.scale_.append(scale)

        return self

    @staticmethod
    def _forward_transformation(val, mean, scale):
        if isinstance(mean, torch.Tensor):
            mean = mean.expand(*val.shape)

        if isinstance(scale, torch.Tensor):
            scale = scale.expand(*val.shape)

            mask = scale != 0
            val = val - mean
            val[mask] = val[mask] / scale[mask]
        else:
            val = (val - mean) / scale

        return val

    @staticmethod
    def _backward_transformation(val, mean, scale):
        if isinstance(mean, torch.Tensor):
            mean = mean.expand(*val.shape)

        if isinstance(scale, torch.Tensor):
            scale = scale.expand(*val.shape)

        return (val * scale) + mean

    def _transformation(self, tmap, transform_fn):
        if not hasattr(self, "keys_") or not hasattr(self, "mean_"):
            raise RuntimeError(f"{self} is not fitted yet.")

        blocks = []

        for key, block in tmap:
            if key not in self.keys_:
                raise RuntimeError(f"{self} has no matching key for key={key}")

            idx = self.keys_.index(key)
            mean = self.mean_[idx]
            scale = self.scale_[idx]

            val = block.values
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).type(torch.float64)
            val = torch.clone(val)

            if self.components_to_sample:
                sdim, cdim, pdim = val.shape
                val = val.reshape(sdim * cdim, pdim)

            val = transform_fn(val, mean, scale)

            if self.components_to_sample:
                val = val.reshape(sdim, cdim, pdim)

            blocks.append(
                TensorBlock(
                    values=val,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

        tmap = TensorMap(
            keys=tmap.keys,
            blocks=blocks,
        )

        return tmap

    def transform(self, tmap):
        return self._transformation(tmap, self._forward_transformation)

    def fit_transform(self, tmap):
        return self.fit(tmap).transform(tmap)

    def inverse_transform(self, tmap):
        return self._transformation(tmap, self._backward_transformation)
