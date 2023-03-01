from torch.utils.data import Dataset

from .utils import load_frames


class SCFDataset(Dataset):
    def __init__(self, frames, focks, ovlps, orbs):
        self.frames = frames
        self.orbs = orbs
        self.focks = focks
        self.ovlps = ovlps

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, _frames):
        if isinstance(_frames, str):
            self._frames = load_frames(_frames)
