import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

# from equistore import Labels
# from equistore import operations as eqop

from .utils import load_frames, load_orbs, fix_pyscf_l1_orbs, fix_pyscf_l1
from .operations import lowdin_orthogonalize
from .hamiltonian import (
    couple_blocks,
    dense_to_blocks,
    #    compute_ham_features,
    #    drop_unused_features,
)


class SCFData:
    def __init__(self, frames, focks, ovlps, orbs, cg, max_frames=None):
        self.max_frames = max_frames
        self.frames = frames
        self.orbs = orbs
        self.focks = focks
        self.ovlps = ovlps
        self.cg = cg

        self.focks_orth = lowdin_orthogonalize(self.focks, self.ovlps)
        self.focks_orth_tmap = dense_to_blocks(self.focks_orth, self.frames, self.orbs)
        self.focks_orth_tmap_coupled = couple_blocks(self.focks_orth_tmap, cg=self.cg)

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, _frames):
        if isinstance(_frames, str):
            # try to load the frames
            self._frames = load_frames(_frames)
        else:
            self._frames = _frames
        self.n_frames = (
            len(self._frames) if self.max_frames is None else self.max_frames
        )
        self._frames = self._frames[: self.n_frames]

    @property
    def orbs(self):
        return self._orbs

    @orbs.setter
    def orbs(self, _orbs):
        if isinstance(_orbs, str):
            # try to load the orbitals
            _orbs = load_orbs(_orbs)
        # fix the l1 order of AOs
        self._orbs = fix_pyscf_l1_orbs(_orbs)

    def _fix_pyscf_l1(self, matrices):
        return torch.stack(
            [
                fix_pyscf_l1(matrix, frame, self.orbs)
                for matrix, frame in zip(matrices, self.frames)
            ]
        )

    def _ensure_torch(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, np.ndarray):
                return torch.from_numpy(tensor)
            else:
                raise RuntimeError(f"tensor is {type(tensor)}, cannot convert to torch")
        else:
            return tensor

    @property
    def focks(self):
        return self._focks

    @focks.setter
    def focks(self, _focks):
        assert hasattr(self, "frames")
        assert hasattr(self, "n_frames")
        assert hasattr(self, "orbs")

        if isinstance(_focks, str):
            _focks = torch.from_numpy(np.load(_focks))

        _focks = self._ensure_torch(_focks)[: self.n_frames]
        self._focks = self._fix_pyscf_l1(_focks)

    @property
    def ovlps(self):
        return self._ovlps

    @ovlps.setter
    def ovlps(self, _ovlps):
        assert hasattr(self, "frames")
        assert hasattr(self, "n_frames")
        assert hasattr(self, "orbs")

        if isinstance(_ovlps, str):
            _ovlps = torch.from_numpy(np.load(_ovlps))

        _ovlps = self._ensure_torch(_ovlps)[: self.n_frames]

        # check that the basis is normalized
        diag = torch.diagonal(_ovlps, dim1=1, dim2=2).detach().cpu().numpy()
        if not np.allclose(diag, 1):
            warnings.warn("AO basis is not normalized. Be careful what you do.")

        self._ovlps = self._fix_pyscf_l1(_ovlps)


class InMemoryDataset(Dataset):
    raise NotImplementedError


class LazyDataset(Dataset):
    raise NotImplementedError
