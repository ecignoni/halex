"""
Run, e.g., as:

$ python samples_replica.py -i SRT_0_remd.pos_0.xyz -o samples_SOAP0_FPS_300K.xyz
"""

from ase.io import read, write
from rascaline import SoapPowerSpectrum
import numpy as np
from argparse import ArgumentParser
from equistore import sum_over_samples
from skmatter.sample_selection import FPS


def cli_parse():
    parser = ArgumentParser()

    opt = parser.add_argument
    opt(
        "-i",
        "--input_replica",
        required=True,
        help="Input xyz corresponding to a __target ensemble__ (i.e., output of i-pi-remdsort)",
    )
    opt("-o", "--output", required=True, help="Name of the output file (xyz)")

    args = parser.parse_args()
    return args, parser


def verbose_action(msg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(msg)
            res = func(*args, **kwargs)
            print("Done!")
            return res

        return wrapper

    return decorator


def soap0_by_sample(descriptor):
    # Create a TensorMap with:
    # `samples = structure (frame), center (i), species_center (ai)
    # properties = neighbor_species_1 (a1), neighbor_species_2 (a2), l, n1, n2
    descriptor = descriptor.keys_to_samples("species_center")
    descriptor = descriptor.keys_to_properties(
        ["species_neighbor_1", "species_neighbor_2"]
    )

    # Since we want to sample __frames__, we need to sum over i and ai
    descriptor = sum_over_samples(
        descriptor, samples_names=["center", "species_center"]
    )
    return descriptor


@verbose_action("Computing 0-SOAP features...")
def compute_soap0(frames):
    HYPER_PARAMETERS = {
        "cutoff": 4.5,  # info coming form Grisafi paper
        "max_radial": 6,
        "max_angular": 4,
        "atomic_gaussian_width": 0.2,  # info coming from Grisafi paper
        "center_atom_weight": 1.0,
        "radial_basis": {
            "Gto": {},
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.5},
        },
    }

    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    descriptor = calculator.compute(frames)
    descriptor = soap0_by_sample(descriptor)
    return np.array(descriptor.block().values)


@verbose_action("Sampling frames with FPS on 0-SOAP features...")
def fps_sample(X):
    selector = FPS(
        n_to_select=1000,  # as in Grisafi/Fabrizio
        progress_bar=True,
        score_threshold=1e-12,
        full=False,
        initialize=0,  # start from the first structure (the one from which the simulation is started)
        random_state=2023,
    )
    selector.fit(X)
    return selector.selected_idx_


if __name__ == "__main__":
    args, _ = cli_parse()

    # Load frames
    frames = read(args.input_replica, ":")

    # Compute 0-SOAP (original SOAP)
    X = compute_soap0(frames)

    if len(frames) != X.shape[0]:
        raise RuntimeError(
            f"number of sample in descriptor X, {X.shape[0]}, is different from number of frames, {len(frames)}"
        )

    # Sample frames with FPS
    sampled_indices = fps_sample(X)

    # shuffle the indices (except from the first)
    # as this is a greedy selector (the most diverse are
    # at the beginning)
    np.random.shuffle(sampled_indices[1:])
    sampled_frames = [frames[i] for i in sampled_indices]

    # Save
    write(args.output, sampled_frames)
