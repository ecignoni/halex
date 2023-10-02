import warnings

import numpy as np
import torch
from metatensor import Labels, TensorBlock, TensorMap
from tqdm import tqdm


class EquivariantPCA:
    def __init__(
        self,
        n_components=None,
        verbose=True,
        key_l_name="spherical_harmonics_l",
    ):
        self.n_components = n_components
        self.verbose = verbose
        self.key_l_name = key_l_name

    @staticmethod
    def _get_mean(values, l):  # noqa
        return 0.0
        # if l == 0:
        #     sums = np.sum(values.detach().numpy(), axis=1)
        #     signs = torch.from_numpy(((sums <= 0) - 0.5) * 2.0)
        #     values = signs[:, None] * values
        #     mean = torch.mean(values, dim=0)
        #     return mean
        # else:
        #     return 0.0

    def _svdsolve(self, X):
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = self._svd_flip(U, Vt)
        eigs = torch.pow(S, 2) / (X.shape[0] - 1)
        return eigs, Vt.T

    @staticmethod
    def _svd_flip(u, v):
        """translated from sklearn implementation"""
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
        return u, v

    def _fit(self, values, l):  # noqa
        nsamples, ncomps, nprops = values.shape
        values = values.reshape(nsamples * ncomps, nprops)

        mean = self._get_mean(values, l)

        eigs, components = self._svdsolve(values - mean)

        return mean, eigs, components

    def fit(self, tmap):
        keys_ = []
        mean_ = []
        explained_variance_ = []
        explained_variance_ratio_ = []
        components_ = []
        retained_components_ = []

        iterator = (
            tqdm(tmap.items(), desc="fitting PCA on each tensormap key")
            if self.verbose
            else tmap
        )
        for key, block in iterator:
            l = key[self.key_l_name]  # noqa

            nsamples, ncomps, nprops = block.values.shape
            mean, eigs, components = self._fit(block.values, l=l)

            eigs_ratio = eigs / torch.sum(eigs)
            eigs_csum = torch.cumsum(eigs_ratio, dim=0)

            if self.n_components is None:
                max_comp = components.shape[1]
                retained = torch.arange(max_comp)

            elif 0 < self.n_components < 1:
                max_comp = (eigs_csum > self.retain_variance).nonzero()[1, 0]
                retained = torch.arange(max_comp)

            elif self.n_components < min(nsamples * ncomps, nprops):
                max_comp = self.n_components
                retained = torch.arange(max_comp)

            # if n_components is too big, do not throw an error but
            # use all the available components
            else:
                warnings.warn("n_components too big: retaining everything")
                max_comp = min(nsamples * ncomps, nprops)
                retained = torch.arange(max_comp)

            eigs = eigs[retained]
            eigs_ratio = eigs_ratio[retained]
            components = components[:, retained]

            keys_.append(tuple(key))
            mean_.append(mean)
            explained_variance_.append(eigs)
            explained_variance_ratio_.append(eigs_ratio)
            components_.append(components)
            retained_components_.append(retained)

        self.keys_ = keys_
        self.mean_ = mean_
        self.explained_variance_ = explained_variance_
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.components_ = components_
        self.retained_components_ = retained_components_

        return self

    def _check_is_fitted(self):
        if not hasattr(self, "components_"):
            raise RuntimeError(f"{self} is not fitted.")

    def transform(self, tmap):
        self._check_is_fitted()

        blocks = []

        iterator = (
            tqdm(tmap.items(), desc="transforming each tensormap key")
            if self.verbose
            else tmap
        )
        tmap_keys = []
        for key, block in iterator:
            try:
                idx = self.keys_.index(tuple(key))
            except ValueError as e:
                if self.verbose:
                    print(str(e))
                continue

            tmap_keys.append(list(key))
            values = block.values.clone()
            nsamples, ncomps, nprops = values.shape
            values = values.reshape(nsamples * ncomps, nprops)

            mean = self.mean_[idx]
            components = self.components_[idx]
            retained = self.retained_components_[idx]
            nretained = len(retained)

            values = torch.matmul(values - mean, components)
            values = values.reshape(nsamples, ncomps, nretained)

            properties = Labels(
                names=["pc"],
                values=np.atleast_2d(np.array([[i] for i in range(nretained)])),
            )

            block = TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            blocks.append(block)

        tmap_keys = Labels(names=tmap.keys.names, values=np.array(tmap_keys))

        return TensorMap(tmap_keys, blocks)

    def inverse_transform(self, tmap):
        self._check_is_fitted()

        blocks = []

        iterator = (
            tqdm(tmap.items(), desc="transforming back each tensormap key")
            if self.verbose
            else tmap
        )
        for key, block in iterator:
            idx = self.keys_.index(key)

            values = block.values.clone()
            nsamples, ncomps, nprops = values.shape
            values = values.reshape(nsamples * ncomps, nprops)

            mean = self.mean_[idx]
            components = self.components_[idx]
            norig = components.shape[0]

            values = torch.matmul(values, components.T) + mean
            values = values.reshape(nsamples, ncomps, norig)

            properties = Labels(
                names=["orig"],
                values=np.atleast_2d(np.array([[i] for i in range(norig)])),
            )

            block = TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            blocks.append(block)

        return TensorMap(tmap.keys, blocks)

    def fit_transform(self, tmap):
        return self.fit(tmap).transform(tmap)

    def save(self, fname):
        self._check_is_fitted()
        tosave = {}
        tosave["keys"] = []
        for key, mean, comp, ret in zip(
            self.keys_, self.mean_, self.components_, self.retained_components_
        ):
            tosave["keys"].append(key)
            tosave[f"{key}_mean"] = np.array(mean)
            tosave[f"{key}_components"] = comp.detach().numpy()
            tosave[f"{key}_retained_components"] = ret.detach().numpy()
        np.savez(fname, **tosave)

    def load(self, fname):
        saved = np.load(fname)
        keys = [tuple(k) for k in saved["keys"]]
        means = [saved[f"{key}_mean"] for key in keys]
        comps = [torch.from_numpy(saved[f"{key}_components"]) for key in keys]
        rets = [torch.from_numpy(saved[f"{key}_retained_components"]) for key in keys]
        self.keys_ = keys
        self.mean_ = means
        self.components_ = comps
        self.retained_components_ = rets
        return self
