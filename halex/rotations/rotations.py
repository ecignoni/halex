import numpy as np
from scipy.spatial.transform import Rotation
from sympy.physics.wigner import wigner_d
from .cgreal import _real2complex


def wigner_d_matrix(l, alpha, beta, gamma):  # noqa: E741
    """Computes a Wigner D matrix
     D^l_{mm'}(alpha, beta, gamma)
    from sympy and converts it to numerical values.
    (alpha, beta, gamma) are Euler angles (radians, ZYZ convention) and l the irrep.
    """
    return np.complex128(wigner_d(l, alpha, beta, gamma))


def rotation_matrix(alpha, beta, gamma):
    """A Cartesian rotation matrix in the appropriate convention
    (ZYZ, implicit rotations) to be consistent with the common Wigner D definition.
    (alpha, beta, gamma) are Euler angles (radians)."""
    return Rotation.from_euler("ZYZ", [alpha, beta, gamma]).as_matrix()


def wigner_d_real(l, alpha, beta, gamma):  # noqa: E741
    """Computes a real-valued Wigner D matrix
     D^l_{mm'}(alpha, beta, gamma)
    (alpha, beta, gamma) are Euler angles (radians, ZYZ convention) and l the irrep.
    Rotates real spherical harmonics by application from the left.
    """

    wd = np.complex128(wigner_d(l, alpha, beta, gamma))
    r2c = _real2complex(l)
    return np.real(np.conjugate(r2c.T @ wd) @ r2c)


def xyz_to_spherical(data, axes=()):
    """
    Converts a vector (or a list of outer products of vectors) from
    Cartesian to l=1 spherical form. Given the definition of real
    spherical harmonics, this is just mapping (y, z, x) -> (-1,0,1)

    Automatically detects which directions should be converted

    data: array
        An array containing the data that must be converted

    axes: array_like
        A list of the dimensions that should be converted. If
        empty, selects all dimensions with size 3. For instance,
        a list of polarizabilities (ntrain, 3, 3) will convert
        dimensions 1 and 2.

    Returns:
        The array in spherical (l=1) form
    """
    shape = data.shape
    # automatically detect the xyz dimensions
    if len(axes) == 0:
        axes = np.where(np.asarray(shape) == 3)[0]
    return np.roll(data, -1, axis=axes)


def spherical_to_xyz(data, axes=()):
    """
    The inverse operation of xyz_to_spherical. Arguments have the
    same meaning, only it goes from l=1 to (x,y,z).
    """
    shape = data.shape
    # automatically detect the l=1 dimensions
    if len(axes) == 0:
        axes = np.where(np.asarray(shape) == 3)[0]
    return np.roll(data, 1, axis=axes)
