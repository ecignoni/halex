from ase.io import read
import os
import numpy as np
from tqdm import tqdm
import time


def write_gaussian_input(frame, filename='tmp.com'):
    with open(filename, 'w') as handle:
        handle.write('%nosave\n')
        handle.write('#p ZINDO nosymm\n')
        handle.write('\n')
        handle.write('Excited state calculation with ZINDO\n')
        handle.write('\n')
        handle.write('0 1\n')
        for atom in frame:
            num, (x, y, z) = atom.number, atom.position
            handle.write('%-10d %10.6f %10.6f %10.6f\n' % (num, x, y, z))
        handle.write('\n')


def crash_if_failed(errcode):
    if errcode != 0:
        raise RuntimeError('Something is wrong with Gaussian')


def check_normal_termination(logfile):
    res = os.popen('grep "Normal termination" tmp.log').read()
    errcode = 1 if len(res) == 0 else 0
    crash_if_failed(errcode)


def get_excitation_energies(logfile):
    num_excens = 3
    res = os.popen("grep 'Excited State' tmp.log | awk '{print $5}'").read().split('\n')[:num_excens]
    res = [float(r) for r in res]
    return res



if __name__ == '__main__':

    atoms = read('atoms.xyz', ':')

    excens = []

    with open('logbook', 'w') as handle:
        pass

    for i, frame in tqdm(enumerate(atoms), desc='ZINDO calc. with Gaussian'):
        write_gaussian_input(frame)

        start = time.perf_counter()
        errcode = os.system('g16 tmp.com')
        elapsed = time.perf_counter() - start

        crash_if_failed(errcode)

        check_normal_termination('tmp.log')

        exc = get_excitation_energies('tmp.log')

        excens.append(exc)

        with open('logbook', 'a') as handle:
            handle.write('Done frame %d / %d in %.2f s\n' % (i+1, len(atoms), elapsed))

    excens = np.array(excens)

    np.save('zindo_excens.npy', excens)
