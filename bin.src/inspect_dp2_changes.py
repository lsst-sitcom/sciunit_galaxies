# Inspect a CDFS (sub)patch

from copy import deepcopy
import math

from astropy.coordinates import SkyCoord
import astropy.units as u
import lsst.daf.butler as dafButler
from lsst.geom import Box2I, degrees, Extent2I, Point2D, Point2I, SpherePoint
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
from lsst.pipe.tasks.prettyPictureMaker import lsstRGB
from lsst.sitcom.sciunit.galaxies.cdfs_hst import get_cutouts_cdfs_hst, scale_cdf_hst_asec
from lsst.sitcom.sciunit.galaxies.euclid import get_cutouts_euclid
from lsst.sitcom.sciunit.galaxies.hst import abs_mag_sol_hst
from lsst.sitcom.sciunit.galaxies.lsst import scale_lsst_asec
from lsst.sitcom.sciunit.galaxies.plotting import plot_external_matches
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


kwargs_ppm = {
    "scaleLumKWargs": {"Q": 8.0, "stretch": 500, "highlight": 0.905882, "shadow": 0.12, "midtone": 0.25},
    "remapBoundsKwargs": {"absMax": 15000},
    "cieWhitePoint": (0.28, 0.28),
    "doLocalContrast": False,
    "scaleColorKWargs": {"maxChroma": 80, "saturation": 0.6},
}

mpl.rcParams.update({"image.origin": "lower", 'font.size': 13})

collection = "u/dtaranu/DM-50135/DP2/v30_0_6_rc1"
butler = dafButler.Butler("/repo/main", collections=collection)

skymap = "lsst_cells_v1"
skymapInfo = butler.get("skyMap", skymap=skymap, collections="skymaps")
