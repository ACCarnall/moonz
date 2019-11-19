from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd
import os
import deepdish as dd

from sklearn.decomposition import PCA
from astropy.io import fits
from spectres import spectres

from .make_model_grid import make_model_grid

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    rank = 0
    size = 1

path = os.path.dirname(os.path.realpath(__file__))


def mpi_split_array(array):
    """ Distributes array elements to cores when using mpi. """
    if size > 1: # If running on more than one core

        n_per_core = array.shape[0]//size

        # How many are left over after division between cores
        remainder = array.shape[0]%size

        if rank == 0:
            if remainder == 0:
                core_array = array[:n_per_core, ...]

            else:
                core_array = array[:n_per_core+1, ...]

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                comm.send(array[start:stop, ...], dest=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                comm.send(array[start:stop, ...], dest=i)

        if rank != 0:
            core_array = comm.recv(source=0)

    else:
        core_array = array

    return core_array


def mpi_combine_array(core_array, total_len):
    """ Combines array sections from different cores. """
    if size > 1: # If running on more than one core

        n_per_core = total_len//size

        # How many are left over after division between cores
        remainder = total_len%size

        if rank != 0:
            comm.send(core_array, dest=0)
            array = None

        if rank == 0:
            array = np.zeros([total_len] + list(core_array.shape[1:]))
            array[:core_array.shape[0], ...] = core_array

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                array[start:stop, ...] = comm.recv(source=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                array[start:stop, ...] = comm.recv(source=i)

        array = comm.bcast(array, root=0)

    else:
        array = core_array

    return array


class batch_fit:

    def __init__(self, IDs, spec_wavs, spec_cube, err_cube, n_components=20,
                 n_train=10000, max_redshift=5., redshift_interval=0.01,
                 run="_", save_correct=False, z_input=None):

        self.IDs = IDs
        self.spec_wavs = spec_wavs
        self.spec_cube = spec_cube
        self.err_cube = err_cube
        self.max_redshift = max_redshift
        self.redshift_interval = redshift_interval
        self.n_components = n_components
        self.run = run
        self.save_correct = save_correct
        self.z_input = z_input

        # Find no of redshift grid points at which models will be fitted
        self.n_grid = int(max_redshift/redshift_interval)+1

        # Load grid of models
        if not os.path.exists(path + "/model_grid.fits"):
            print("Making model grid, this will take ~10-20 minutes the"
                   + " first time you run the code...")
            make_model_grid()

        data_file = fits.open(path + "/model_grid.fits")

        self.spec_grid = data_file[1].data[:n_train, :]

        # Load wavelengths for model grid
        self.pc_wavs = data_file[2].data

    def fit(self):

        # Set up zvals - grid points to be distributed among cores
        zvals = np.linspace(0., self.max_redshift, self.n_grid)

        core_zvals = mpi_split_array(zvals)

        core_all_chisq = np.zeros((self.spec_cube.shape[0],
                                   core_zvals.shape[0]))

        for i in range(core_zvals.shape[0]):

            z = core_zvals[i]  # Redshift at which to fit
            print("Fitting z =", z)
            # Resample (redshifted) model grid to desired wavelengths
            spec_grid_res = spectres(self.spec_wavs, self.pc_wavs*(1.+z),
                                     self.spec_grid)

            # Do PCA on model grid at the chosen redshift
            pca = PCA(n_components=self.n_components)
            pca.fit(spec_grid_res)

            # Do principal component decomposition of observed spectra
            coefs = np.dot(pca.components_, self.spec_cube.T)

            # Calculate chi-squared values for best PCA decomposition
            # Could be made into an array operation, not limiting step
            for j in range(self.spec_cube.shape[0]):
                best_model = np.sum(coefs[:, j]*pca.components_.T, axis=1)
                resid = (best_model - self.spec_cube[j, :])/self.err_cube[j, :]
                chisq = np.sum(resid**2)
                dof = float(self.spec_wavs.shape[0] - self.n_components)
                core_all_chisq[j, i] = chisq/dof
                """
                if self.save_correct and ((z - self.z_input[j])**2 < 0.01**2):
                    best_model = np.sum(coefs[:, j]*pca.components_.T, axis=1)
                    spec = np.c_[self.spec_wavs, best_model,
                                 self.spec_cube[j, :], self.err_cube[j, :]]

                    np.savetxt("best_model/" + self.IDs[j] + ".txt", spec)
                """
        all_chisq = mpi_combine_array(core_all_chisq.T, self.n_grid).T

        if rank == 0:
            dd.io.save("all_chisq_" + self.run + ".h5", all_chisq)

            best_z = np.argmin(all_chisq, axis=1)*self.redshift_interval
            cat = pd.DataFrame(np.c_[self.IDs, best_z],
                               columns=["#ID", "z_best"])

            cat.to_csv("best_z_" + self.run + ".txt", sep="\t", index=False)
