from __future__ import print_function, division, absolute_import

import numpy as np
import os

from astropy.io import fits

path = os.getcwd()#os.path.dirname(os.path.realpath(__file__))


def make_model_grid(n_train=25000, max_redshift=5.5):

    import bagpipes as pipes

    dblplaw = {}
    dblplaw["massformed"] = 1.
    dblplaw["metallicity"] = (0.2, 2.5)
    dblplaw["metallicity_prior"] = "log_10"
    dblplaw["alpha"] = (0.01, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta"] = (0.01, 1000.)
    dblplaw["beta_prior"] = "log_10"
    dblplaw["tau"] = (0.1, 15.)

    nebular = {}
    nebular["logU"] = -3.

    dust = {}
    dust["type"] = "Calzetti"
    dust["eta"] = 1.
    dust["Av"] = (0., 1.)

    fit_instructions = {}
    fit_instructions["dblplaw"] = dblplaw
    fit_instructions["nebular"] = nebular
    fit_instructions["dust"] = dust
    fit_instructions["veldisp"] = (1., 400.)
    fit_instructions["veldisp_prior"] = "log_10"
    fit_instructions["redshift"] = 0.
    fit_instructions["t_bc"] = 0.01

    pc_wavs = np.arange(6400./(1. + max_redshift), 18100., 1)

    prior = pipes.fitting.check_priors(fit_instructions, spec_wavs=pc_wavs,
                                       n_draws=n_train)

    prior.get_advanced_quantities()

    spec_grid = prior.samples["spectrum"]

    mask = np.max(spec_grid, axis=1).astype(bool)

    spec_grid = spec_grid[mask, :]

    for i in range(spec_grid.shape[0]):
        spec_grid[i, :] /= np.mean(spec_grid[i, :])

    hdulist = fits.HDUList(hdus=[fits.PrimaryHDU(),
                                 fits.ImageHDU(name="grid", data=spec_grid),
                                 fits.ImageHDU(name="wavs", data=pc_wavs)])

    if os.path.exists(path + "/model_grid.fits"):
        os.system("rm " + path + "/model_grid.fits")

    hdulist.writeto(path + "/model_grid.fits")
