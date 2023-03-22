from __future__ import annotations
from typing import Dict, List, Union, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from equistore import Labels, TensorMap
from equistore import operations as eqop

from .utils import (
    load_frames,
    load_orbs,
    fix_pyscf_l1_orbs,
    fix_pyscf_l1,
    get_ao_labels,
    atomic_number2symbol,
)
from .operations import lowdin_orthogonalize
from .hamiltonian import (
    couple_blocks,
    dense_to_blocks,
)
from .popan import (
    batched_orthogonal_lowdin_population,
    batched_orthogonal_lowdinbyMO_population,
)
from .model_selection import train_test_split


class SCFData:
    def __init__(
        self,
        frames: Union[str, List[Atoms]],  # noqa
        focks: Union[str, torch.Tensor, np.ndarray],
        ovlps: Union[str, torch.Tensor, np.ndarray],
        orbs: Union[str, Dict[int, List]],
        cg: "ClebshGordanReal",  # noqa
        max_frames: int = None,
        skip_first_n: int = 0,
        indices: Union[torch.Tensor, np.ndarray] = None,
        nelec_dict: Dict[str, float] = None,
    ) -> None:
        self.max_frames = max_frames
        self.skip_first_n = skip_first_n
        self.indices = indices
        self.frames = frames
        self.orbs = orbs
        self.focks = focks
        self.ovlps = ovlps
        self.cg = cg
        self.nelec_dict = nelec_dict

        self.focks_orth = lowdin_orthogonalize(self.focks, self.ovlps)
        self.focks_orth_tmap = dense_to_blocks(self.focks_orth, self.frames, self.orbs)
        self.focks_orth_tmap_coupled = couple_blocks(self.focks_orth_tmap, cg=self.cg)

        self.ao_labels = get_ao_labels(self.orbs, self.frames[0].numbers)
        self.mo_energy, self.mo_coeff_orth = torch.linalg.eigh(self.focks_orth)
        self.lowdin_charges, _ = batched_orthogonal_lowdin_population(
            focks_orth=self.focks_orth,
            nelec_dict=self.nelec_dict,
            ao_labels=self.ao_labels,
        )
        self.lowdin_charges_byMO, _ = batched_orthogonal_lowdinbyMO_population(
            focks_orth=self.focks_orth,
            nelec_dict=self.nelec_dict,
            ao_labels=self.ao_labels,
        )
        self.mo_occ = self._get_mo_occupancy()
        self.atom_pure_symbols = self.frames[0].get_chemical_symbols()
        self.natm = len(self.atom_pure_symbols)

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, _frames):
        if isinstance(_frames, str):
            # try to load the frames
            _frames = load_frames(_frames)

        self._frames = _frames

        if self.indices is None:
            # we are not given indices: load with start/stop
            self._max_frames = (
                len(self._frames) if self.max_frames is None else self.max_frames
            )
            self._frames = self._frames[self.skip_first_n : self._max_frames]

        else:
            # we are given indices: load them
            self._frames = [self._frames[i] for i in self.indices]

        self.n_frames = len(self._frames)

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

        _focks = self._ensure_torch(_focks)

        if self.indices is None:
            _focks = _focks[self.skip_first_n : self._max_frames]

        else:
            _focks = _focks[self.indices]

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

        _ovlps = self._ensure_torch(_ovlps)

        if self.indices is None:
            _ovlps = _ovlps[self.skip_first_n : self._max_frames]
        else:
            _ovlps = _ovlps[self.indices]

        # check that the basis is normalized
        diag = torch.diagonal(_ovlps, dim1=1, dim2=2).detach().cpu().numpy()
        if not np.allclose(diag, 1):
            raise ValueError("AO basis is not normalized.")

        self._ovlps = self._fix_pyscf_l1(_ovlps)

    @property
    def nelec_dict(self):
        return self._nelec_dict

    @nelec_dict.setter
    def nelec_dict(self, _nelec_dict):
        assert hasattr(self, "frames")
        if _nelec_dict is None:
            _nelec_dict = {
                atomic_number2symbol[n]: float(n) for n in self.frames[0].numbers
            }
        self._nelec_dict = _nelec_dict

    def _get_mo_occupancy(self):
        symbols = self.frames[0].get_chemical_symbols()
        nel = sum([self.nelec_dict[sym] for sym in symbols])
        nmo = self.focks_orth.shape[-1]
        mo_occ = torch.zeros(nmo, dtype=int)
        for i in range(int(nel / 2)):
            mo_occ[i] = 2.0
        return mo_occ

    def train_test_split(self, train_size=0.8):
        (
            train_frames,
            test_frames,
            train_focks,
            test_focks,
            train_ovlps,
            test_ovlps,
        ) = train_test_split(
            self.frames,
            self.focks,
            self.ovlps,
            n_frames=self.n_frames,
            train_size=train_size,
        )

        train_data = SCFData(
            frames=train_frames,
            focks=train_focks,
            ovlps=train_ovlps,
            orbs=self.orbs,
            cg=self.cg,
            max_frames=len(train_frames),
            nelec_dict=self.nelec_dict,
        )

        test_data = SCFData(
            frames=test_frames,
            focks=test_focks,
            ovlps=test_ovlps,
            orbs=self.orbs,
            cg=self.cg,
            max_frames=len(test_frames),
            nelec_dict=self.nelec_dict,
        )

        return train_data, test_data


def slice_list(llist: List, indices: np.ndarray) -> List:
    return [llist[i] for i in indices]


def slice_tensor(tensor: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    return tensor[indices]


def slice_tmap(tmap: TensorMap, indices: np.ndarray) -> TensorMap:
    return eqop.slice(
        tmap,
        samples=Labels(
            names=["structure"], values=np.atleast_1d(indices).reshape(-1, 1)
        ),
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
    def __init__(
        self, n_samples: int, *data: Any, batch_size: int = 64, **metadata: Any
    ) -> None:
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
