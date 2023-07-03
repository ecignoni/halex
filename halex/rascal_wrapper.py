from typing import List
import copy
import numpy as np
import torch

import warnings

import ase
from rascal.representations import SphericalExpansion

from equistore import Labels, TensorBlock, TensorMap


class RascalSphericalExpansion:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames: List[ase.Atoms], global_species: List[int] = None) -> TensorMap:  # noqa: C901
        # Step 1: compute spherical expansion with librascal
        hypers = copy.deepcopy(self._hypers)

        # if global_species is not given, infer it from the ase objects
        if global_species is None:
            global_species = list(
                map(int, np.unique(np.concatenate([f.numbers for f in frames])))
            )

        hypers["global_species"] = global_species
        hypers["expansion_by_species_method"] = "user defined"
        # assert
        calculator = SphericalExpansion(**hypers)
        manager = calculator.transform(frames)
        data = manager.get_features(calculator)
        gradients = manager.get_features_gradient(calculator)
        info = manager.get_representation_info()
        grad_info = manager.get_gradients_info()

        # Step 2: move data around to follow the storage convention
        keys = Labels(
            names=["spherical_harmonics_l", "species_center", "species_neighbor"],
            values=np.array(
                [
                    [l, species_center, species_neighbor]
                    for l in range(hypers["max_angular"] + 1)
                    for species_center in global_species
                    for species_neighbor in global_species
                ],
                dtype=np.int32,
            ),
        )

        properties = Labels(
            names=["n"],
            values=np.array([[n] for n in range(hypers["max_radial"])], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(hypers["max_angular"] + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        data = data.reshape(
            data.shape[0], len(global_species), hypers["max_radial"], -1
        )
        gradients = gradients.reshape(
            gradients.shape[0], len(global_species), hypers["max_radial"], -1
        )

        global_to_per_structure_atom_id = []
        for frame in frames:
            for i in range(len(frame)):
                global_to_per_structure_atom_id.append(i)

        blocks = []
        for l, species_center, species_neighbor in keys:
            species_neighbor_i = global_species.index(species_neighbor)
            species_center_mask = np.where(info[:, 2] == species_center)[0]
            block_data = data[species_center_mask, species_neighbor_i, :, lm_slices[l]]
            block_data = block_data.swapaxes(1, 2)

            samples = Labels(
                names=["structure", "center"],
                values=np.copy(info[species_center_mask, :2]).astype(np.int32),
            )
            spherical_component = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            block_gradients = None
            gradient_samples = None
            if hypers["compute_gradients"]:
                warnings.warn(
                    "numpy/forward gradients are currently broken with librascal,"
                    "please use rascaline instead"
                )
                # the code below is missing some of the gradient sample, making
                # the forces not match the energy in a finite difference test

                gradient_samples = []
                block_gradients = []
                for sample_i, (structure, center) in enumerate(samples):
                    gradient_mask = np.logical_and(
                        np.logical_and(
                            grad_info[:, 0] == structure,
                            grad_info[:, 1] == center,
                        ),
                        grad_info[:, 4] == species_neighbor,
                    )

                    for grad_index in np.where(gradient_mask)[0]:
                        start = 3 * grad_index
                        stop = 3 * grad_index + 3
                        block_gradient = gradients[
                            start:stop, species_neighbor_i, :, lm_slices[l]
                        ]
                        block_gradient = block_gradient.swapaxes(1, 2)
                        block_gradient = block_gradient.reshape(
                            -1, 3, block_gradient.shape[1], block_gradient.shape[2]
                        )
                        if np.linalg.norm(block_gradient) == 0:
                            continue

                        block_gradients.append(block_gradient)

                        structure = grad_info[grad_index, 0]
                        neighbor = global_to_per_structure_atom_id[
                            grad_info[grad_index, 2]
                        ]
                        gradient_samples.append((sample_i, structure, neighbor))

                if len(gradient_samples) != 0:
                    block_gradients = np.concatenate(block_gradients)
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom"],
                        values=np.vstack(gradient_samples).astype(np.int32),
                    )
                else:
                    block_gradients = np.zeros(
                        (0, 3, spherical_component.shape[0], properties.shape[0])
                    )
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom"],
                        values=np.zeros((0, 3), dtype=np.int32),
                    )

            # reset atom index (librascal uses a global atom index)
            samples = Labels(
                names=["structure", "center"],
                values=np.array(
                    [
                        (structure, global_to_per_structure_atom_id[center])
                        for structure, center in samples
                    ],
                    dtype=np.int32,
                ),
            )

            block = TensorBlock(
                values=np.copy(block_data),
                samples=samples,
                components=[spherical_component],
                properties=properties,
            )

            spatial_component = Labels(
                names=["gradient_direction"],
                values=np.array([[0], [1], [2]], dtype=np.int32),
            )
            gradient_components = [spatial_component, spherical_component]

            if block_gradients is not None:
                block.add_gradient(
                    "positions",
                    block_gradients,
                    gradient_samples,
                    gradient_components,
                )

            blocks.append(block)

        return TensorMap(keys, blocks)


class RascalPairExpansion:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames: List[ase.Atoms], global_species: List[int] = None) -> TensorMap:
        # save here global_species under another name (same interface but used 
        # differently inside the function)
        all_species = global_species

        # Step 1: compute spherical expansion with librascal
        hypers = copy.deepcopy(self._hypers)
        if hypers["compute_gradients"]:
            raise Exception("Pair expansion with gradient is not implemented")

        max_atoms = max([len(f) for f in frames])
        global_species = list(range(max_atoms))

        hypers["global_species"] = global_species
        hypers["expansion_by_species_method"] = "user defined"
        # if 'optimization' in hypers.keys():
        #     if 'RadialDimReduction' in hypers['optimization'].keys():
        #         hypers['optimization']['RadialDimReduction']['projection_matrices'] = \
        #            {i: self._hypers['optimization']['RadialDimReduction']['projection_matrices'][sp]
        #             for i, sp in enumerate(frames[0].numbers)}

        ijframes = []

        # if all_species (the global_species provided) is None, infer it from the ase objects
        if all_species is None:
            all_species = np.unique(np.hstack([f.numbers for f in frames]))

        for f in frames:
            ijf = f.copy()
            ijf.numbers = global_species[: len(f)]
            ijframes.append(ijf)

        calculator = SphericalExpansion(**hypers)

        # Step 2: move data around to follow the storage convention
        keys = Labels(
            names=["spherical_harmonics_l", "species_center", "species_neighbor"],
            values=np.array(
                [
                    [l, ai, aj]
                    for l in range(hypers["max_angular"] + 1)
                    for ai in all_species
                    for aj in all_species
                ],
                dtype=np.int32,
            ),
        )

        lm_slices = []
        start = 0
        for l in range(hypers["max_angular"] + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        data = []
        samples = []
        species_pairs = []
        for i, (ijf, f) in enumerate(zip(ijframes, frames)):
            idata = (
                calculator.transform(ijf)
                .get_features(calculator)
                .reshape(len(ijf), max_atoms, hypers["max_radial"], -1)
            )
            nonzero = np.where((idata**2).sum(axis=(2, 3)) > 1e-20)
            data.append(
                idata[nonzero[0], nonzero[1]].reshape(
                    len(nonzero[0]), hypers["max_radial"], -1
                )
            )
            species_pairs.append(
                np.asarray(
                    [f.numbers[nonzero[0]], f.numbers[nonzero[1]]], dtype=np.int32
                ).T
            )
            samples.append(np.asarray([nonzero[0] * 0 + i, nonzero[0], nonzero[1]]).T)

        species_pairs = np.vstack(species_pairs)
        data = np.concatenate(data)
        samples = Labels(
            names=["structure", "center", "neighbor"],
            values=np.concatenate(samples).astype(np.int32),
        )

        blocks = []
        for l, a1, a2 in keys:
            selected_samples = np.where(
                (species_pairs[:, 0] == a1) & (species_pairs[:, 1] == a2)
            )[0]
            block_data = data[selected_samples, ..., lm_slices[l]]
            block_data = block_data.swapaxes(1, 2)

            component = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            properties = Labels(
                names=["species_neighbor", "n"],
                values=np.array(
                    [[a2, n] for n in range(hypers["max_radial"])], dtype=np.int32
                ),
            )

            block = TensorBlock(
                values=np.copy(block_data),
                samples=samples[selected_samples],
                components=[component],
                properties=properties,
            )

            blocks.append(block)

        return TensorMap(keys, blocks)


class SphericalExpansionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, calculator, positions, cell, numbers):
        assert isinstance(positions, torch.Tensor)
        assert isinstance(cell, torch.Tensor)
        assert isinstance(numbers, torch.Tensor)

        frame = ase.Atoms(
            numbers=numbers.numpy(), cell=cell.numpy(), positions=positions.numpy()
        )
        manager = calculator.transform(frame)
        descriptor = manager.get_features(calculator)

        samples = manager.get_representation_info()

        grad_descriptor = manager.get_features_gradient(calculator)
        grad_info = manager.get_gradients_info()
        ctx.save_for_backward(
            torch.tensor(grad_descriptor),
            torch.tensor(grad_info),
        )

        return samples, torch.tensor(descriptor)

    @staticmethod
    def backward(ctx, grad_samples, grad_output):
        grad_spherical_expansion, grad_info = ctx.saved_tensors

        grad_calculator = grad_positions = grad_cell = grad_numbers = None

        if ctx.needs_input_grad[0]:
            raise ValueError("can not compute gradients w.r.t. calculator")
        if ctx.needs_input_grad[1]:
            if grad_spherical_expansion.shape[1] == 0:
                raise ValueError(
                    "missing gradients, please set `compute_gradients` to True"
                )

            grad_positions = torch.zeros((grad_output.shape[0], 3))
            for sample, (_, center_i, neighbor_i, *_) in enumerate(grad_info):
                for spatial_i in range(3):
                    sample_i = 3 * sample + spatial_i
                    grad_positions[neighbor_i, spatial_i] += torch.dot(
                        grad_output[center_i, :], grad_spherical_expansion[sample_i, :]
                    )

        if ctx.needs_input_grad[2]:
            raise ValueError("can not compute gradients w.r.t. cell")
        if ctx.needs_input_grad[2]:
            raise ValueError("can not compute gradients w.r.t. atomic numbers")

        return grad_calculator, grad_positions, grad_cell, grad_numbers


class TorchFrame:
    def __init__(self, frame: ase.Atoms, requires_grad: bool):
        self.positions = torch.tensor(frame.positions, requires_grad=requires_grad)
        self.species = torch.tensor(frame.numbers)
        self.cell = torch.tensor(frame.cell[:])

    def __len__(self):
        return self.species.shape[0]


class RascalSphericalExpansionTorch:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames: List[TorchFrame]) -> TensorMap:
        # Step 1: compute spherical expansion with librascal
        hypers = copy.deepcopy(self._hypers)
        global_species = list(
            map(int, np.unique(np.concatenate([f.species.numpy() for f in frames])))
        )

        hypers["global_species"] = global_species
        hypers["expansion_by_species_method"] = "user defined"

        calculator = SphericalExpansion(**hypers)

        all_spx = []
        all_info = []
        for i, frame in enumerate(frames):
            info, spx = SphericalExpansionAutograd.apply(
                calculator, frame.positions, frame.cell, frame.species
            )

            all_spx.append(spx)

            # set the structure id
            info[:, 0] = i
            all_info.append(info)

        data = torch.vstack(all_spx)
        info = np.vstack(all_info)

        # Step 2: move data around to follow the storage convention
        keys = Labels(
            names=["spherical_harmonics_l", "species_center", "species_neighbor"],
            values=np.array(
                [
                    [l, species_center, species_neighbor]
                    for l in range(hypers["max_angular"] + 1)
                    for species_center in global_species
                    for species_neighbor in global_species
                ],
                dtype=np.int32,
            ),
        )

        properties = Labels(
            names=["n"],
            values=np.array([[n] for n in range(hypers["max_radial"])], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(hypers["max_angular"] + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        data = data.reshape(
            data.shape[0], len(global_species), hypers["max_radial"], -1
        )

        blocks = []
        for l, species_center, species_neighbor in keys:
            species_neighbor_i = global_species.index(species_neighbor)
            species_center_mask = np.where(info[:, 2] == species_center)[0]
            block_data = data[species_center_mask, species_neighbor_i, :, lm_slices[l]]
            block_data = block_data.swapaxes(1, 2)

            samples = Labels(
                names=["structure", "center"],
                values=np.copy(info[species_center_mask, :2]).astype(np.int32),
            )
            component = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            blocks.append(
                TensorBlock(
                    values=block_data,
                    samples=samples,
                    components=[component],
                    properties=properties,
                )
            )

        return TensorMap(keys, blocks)
