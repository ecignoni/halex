import numpy as np
import torch
from tqdm import tqdm

from equistore import Labels, TensorBlock, TensorMap


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
            tqdm(tmap, desc="fitting PCA on each tensormap key")
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

            eigs = eigs[retained]
            eigs_ratio = eigs_ratio[retained]
            components = components[:, retained]

            keys_.append(key)
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
            tqdm(tmap, desc="transforming each tensormap key") if self.verbose else tmap
        )
        for key, block in iterator:
            idx = self.keys_.index(key)

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

        return TensorMap(tmap.keys, blocks)

    def inverse_transform(self, tmap):
        self._check_is_fitted()

        blocks = []

        iterator = (
            tqdm(tmap, desc="transforming back each tensormap key")
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


# if __name__ == "__main__":
#     from utils.rotations import rotation_matrix, wigner_d_real
#     from torch_cg import ClebschGordanReal
#     from torch_utils import (
#         load_frames,
#         load_orbs,
#         load_hamiltonians,
#         compute_ham_features,
#         tensormap_as_torch,
#     )
#
#     def compute_features():
#         n_frames = 50
#         frames = load_frames(
#             "../data/hamiltonian/water-hamiltonian/water_coords_1000.xyz",
#             n_frames=n_frames,
#         )
#         orbs = load_orbs("../data/hamiltonian/water-hamiltonian/water_orbs.json")
#         hams = load_hamiltonians(
#             "../data/hamiltonian/water-hamiltonian/water_saph_orthogonal.npy",
#             n_frames=n_frames,
#         )
#
#         cg = ClebschGordanReal(4)
#
#         rascal_hypers = {
#             "interaction_cutoff": 3.5,
#             "cutoff_smooth_width": 0.5,
#             "max_radial": 8,
#             "max_angular": 4,
#             "gaussian_sigma_constant": 0.2,
#             "gaussian_sigma_type": "Constant",
#             "compute_gradients": False,
#         }
#
#         feats = compute_ham_features(rascal_hypers, frames, cg)
#         feats = tensormap_as_torch(feats)
#
#         return feats
#
#     def test_rotation_equivariance_pca():
#         """
#         Equivariance test: f(ŜA) = Ŝ(f(A))
#         Here the operation is a rotation in 3D space: Ŝ = R
#         """
#         # rotation angles, ZYZ
#         alpha = np.pi / 3
#         beta = np.pi / 3
#         gamma = np.pi / 4
#         R = rotation_matrix(alpha, beta, gamma).T
#
#         feats = compute_features()
#
#         epca = EquivariantPCA(verbose=False).fit(feats)
#
#         alpha, beta, gamma = np.pi / 3, np.pi / 3, np.pi / 4
#         R = rotation_matrix(alpha, beta, gamma).T
#
#         A = frames[0].copy()
#         RA = frames[0].copy()
#         RA.positions = RA.positions @ R
#         RA.cell = RA.cell @ R
#
#         f_A_unprocessed = tensormap_as_torch(
#             compute_ham_features(rascal_hypers, [A], cg)
#         )
#         f_RA_unprocessed = tensormap_as_torch(
#             compute_ham_features(rascal_hypers, [RA], cg)
#         )
#
#         f_A = epca.transform(f_A_unprocessed)
#         f_RA = epca.transform(f_RA_unprocessed)
#
#         for (key, block), (_, rotated_block) in zip(f_A, f_RA):
#             l = int(key["spherical_harmonics_l"])
#             D = wigner_d_real(l, alpha, beta, gamma)
#             values = block.values
#             rotated_values = np.einsum("nm,smp->snp", D, values)
#             norm = torch.linalg.norm(rotated_block.values - rotated_values)
#             assert norm < 1e-17, f"mismatch for key={key}, norm = {norm}"
#
#     def test_inverse_transform():
#         """
#         Test if inverse transforming the pca reduced features
#         gives the original features
#         """
#         feats = compute_features()
#         epca = EquivariantPCA(n_components=None, verbose=False).fit(feats)
#         T_feats = epca.transform(feats)
#         rec_feats = epca.inverse_transform(T_feats)
#
#         for (key, block), (_, rec_block) in zip(feats, rec_feats):
#             values = block.values
#             rec_values = rec_block.values
#             norm = torch.linalg.norm(values - rec_values)
#             assert norm < 1e-17, f"mismatch for key={key}, norm = {norm}"
#
#     test_rotation_equivariance_pca()
#     test_inverse_transform()
