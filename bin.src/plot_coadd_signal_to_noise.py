# Make plots of the chi distribution in coadds
# Inspection of a few patches shows it's close to Gaussian with the expected
# positive excess.
# Patches with negative excesses tended to have those pixels randomly
# distributed in regions with fewer input visits

import matplotlib.pyplot as plt
import numpy as np
import scipy

from lsst.afw.detection.utils import footprintsToNumpy
from lsst.meas.extensions.scarlet.io import updateCatalogFootprints
from lsst.utils.plotting import (
    get_multiband_plot_colors, get_multiband_plot_linestyles, make_figure, set_rubin_plotstyle,
)

set_rubin_plotstyle()


def plot_coadd_variance(
    butler, skymap, tract, patch, bands, bins=None, xlim=None, ylim=None, interactive=True,
    linestyles=None, colors=None,
):
    if xlim is None:
        xlim = (-5, 5)
    if ylim is None:
        ylim = (1e-7, 1)
    if bins is None:
        bins = np.linspace(xlim[0], xlim[1], 101)
    if linestyles is None:
        linestyles = get_multiband_plot_linestyles()
    if colors is None:
        colors = get_multiband_plot_colors()

    fig, ax = plt.subplots() if interactive else make_figure()
    erf_bins = scipy.special.erf(bins / np.sqrt(2))/2.
    ax.stairs(
        (erf_bins[1:] - erf_bins[:-1])/(bins[1:] - bins[:-1]), bins, color="gray",
        label="σ=1, med./σ/σ(x<med.)/σ(x>med.)",
    )

    for band in bands:
        coadd = butler.get("deep_coadd", skymap=skymap, tract=tract, patch=patch, band=band)
        meas = butler.get("object_unforced_measurement", skymap=skymap, tract=tract, patch=patch, band=band)
        models_scarlet = butler.get(
            "object_scarlet_models", skymap=skymap, tract=tract, patch=patch, band=band,
        )
        updateCatalogFootprints(
            modelData=models_scarlet,
            catalog=meas,
            band="r",
            imageForRedistribution=coadd,
            removeScarletData=True,
            updateFluxColumns=False,
        )
        fp = footprintsToNumpy(meas, shape=coadd.image.array.shape, xy0=coadd.getXY0())
        sn = coadd.image.array[~fp]/np.sqrt(coadd.variance.array[~fp])

        within = sn[(sn > xlim[0]) & (sn < xlim[1])]
        med = np.nanmedian(within)
        std_cond = tuple(
            np.sqrt(np.sum((sn[cond]-med)**2)/np.sum(cond))
            for cond in ((sn > xlim[0]) & (sn < med), (sn < xlim[1]) & (sn > med))
        )
        std = np.nanstd(within)
        label = f"{band} {med:.2e}/{std:.3f}/{std_cond[0]:.3f}/{std_cond[1]:.3f}"

        ax.hist(
            sn, bins=bins, log=True, histtype="step", density=True,
            label=label, color=colors[band], linestyle=linestyles[band],
        )
        ax.set_xlabel("Pixel S/N")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc="lower center")

    fig.suptitle(f"deep_coadd tract={tract} patch={patch} non-footprint pixels")
    fig.tight_layout()
    return fig, ax
