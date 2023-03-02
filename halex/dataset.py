from __future__ import annotations
from typing import Dict, List, Union, Tuple, Any

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from equistore import Labels, TensorMap
from equistore import operations as eqop

from .utils import load_frames, load_orbs, fix_pyscf_l1_orbs, fix_pyscf_l1
from .operations import lowdin_orthogonalize
from .hamiltonian import (
    couple_blocks,
    dense_to_blocks,
)


class SCFData:
    def __init__(
        self,
        frames: Union[str, List[Atoms]],  # noqa
        focks: Union[str, torch.Tensor, np.ndarray],
        ovlps: Union[str, torch.Tensor, np.ndarray],
        orbs: Union[str, Dict[int, List]],
        cg: "ClebshGordanReal",  # noqa
        max_frames: int = None,
    ) -> None:
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


def slice_list(llist: List, indices: np.ndarray) -> List:
    return [llist[i] for i in indices]


def slice_tensor(tensor: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    return tensor[indices]


def slice_tmap(tmap: TensorMap, indices: np.ndarray) -> TensorMap:
    return eqop.slice(
        tmap, samples=Labels(names=["structure"], values=indices.reshape(-1, 1))
    )


class InMemoryDataset(Dataset):
    def __init__(
        self, n_samples: int, *data: List[Union[List, torch.Tensor, TensorMap]]
    ) -> None:
        self.n_samples = n_samples
        self.data = data
        self._setup_slicing_fns()

    def _setup_slicing_fns(self):
        fns = []
        for d in self.data:
            if isinstance(d, list):
                fns.append(slice_list)
            elif isinstance(d, torch.Tensor):
                fns.append(slice_tensor)
            elif isinstance(d, TensorMap):
                fns.append(slice_tmap)
            else:
                raise ValueError(f"one element of data has a wrong type: {type(d)}")
        self._slicing_fns = fns

    def get_indices(self, batch_size: int) -> List[torch.Tensor]:
        indices = torch.arange(self.n_samples)
        batches = [
            indices[i * batch_size : (i + 1) * batch_size]
            for i in range(self.n_samples // batch_size)
        ]
        if batches[-1][-1] != indices[-1]:
            n = int(batches[-1][-1])
            batches += [indices[n + 1 :]]
        return batches

    def __len__(self):
        return self.n_samples

    def __getitem__(
        self, idx: Union[int, torch.Tensor, List[int], np.ndarray]
    ) -> Tuple:
        idx = np.atleast_1d(idx)
        out = []
        for slicing_fn, d in zip(self._slicing_fns, self.data):
            out.append(slicing_fn(d, indices=idx))
        return tuple(out)


class BatchedMemoryDataset(Dataset):
    def __init__(self, n_samples, *data, batch_size=64, **metadata):
        self.n_samples = n_samples
        self.batch_size = batch_size
        for key, value in metadata.items():
            setattr(self, key, value)
        self._setup_batches(data)

    def _setup_batches(self, data: List[Any]) -> None:
        indices = self.get_indices(self.batch_size)
        batches = []
        for idx in indices:
            batch = []
            for d in data:
                if isinstance(d, list):
                    batch.append(slice_list(d, indices=idx))
                elif isinstance(d, torch.Tensor):
                    batch.append(slice_tensor(d, indices=idx))
                elif isinstance(d, TensorMap):
                    batch.append(slice_tmap(d, indices=idx))
            batches.append(tuple(batch))
        self.batch_indices = indices
        self.n_batches = len(indices)
        self.batches = batches

    def get_indices(self, batch_size: int) -> List[torch.Tensor]:
        indices = torch.arange(self.n_samples)
        batches = [
            indices[i * batch_size : (i + 1) * batch_size]
            for i in range(self.n_samples // batch_size)
        ]
        if batches[-1][-1] != indices[-1]:
            n = int(batches[-1][-1])
            batches += [indices[n + 1 :]]
        return batches

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx: int) -> Tuple:
        return self.batches[idx]


class LazyDataset(Dataset):
    def __init__(self):
        raise NotImplementedError
