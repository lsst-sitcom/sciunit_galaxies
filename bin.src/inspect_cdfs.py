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

ra_gal, dec_gal = 53.12277768, -27.73640709

# gri are the easiest bands to make aesthetically pleasing images from
# Using all 6 bands may be better for understanding detection issues since
# there will be single-band detections (real ones probably mostly in y)
use_gri = False
if use_gri:
    bands_lsst_rgb = (("y", "z"), ("i", "r"), ("g", "u"))
    # TODO: Make some kwargs_ppm changes
else:
    bands_lsst_rgb = (("i",), ("r",), ("g",))

bands_lsst = tuple(band for bands in bands_lsst_rgb for band in bands)

bands_hst = ("F775W", "F606W", "F435W")

weight_mean = np.mean([bands_weights_lsst[band] for band in bands_lsst])
for band in bands_lsst:
    bands_weights_lsst[band] /= weight_mean

scale_ratio_hst = scale_lsst_asec/scale_cdf_hst_asec

coord = SpherePoint(ra_gal, dec_gal, degrees)
# This size should be small enough to inspect the image without too much
# zooming in and out. Note that this is the number of Rubin pixels
# and Euclid will be 2x and HST ~7x for 0.03" mosaics.
cutout_size = Extent2I(400, 400)

# These are only valid for the 400x400 cutout size
zooms_patches = {
    34: {
        (5, 6): (
            ((53.126669, 53.121621), (-27.735940, -27.738999)),
            ((53.128285, 53.124553), (-27.749245, -27.751794)),
            ((53.122157, 53.117033), (-27.740190, -27.744447)),
        )
    }
} if (cutout_size == Extent2I(400, 400)) else {}

collection = "u/dtaranu/DM-50091/v29_0_0_rc6/match"
butler = dafButler.Butler("/repo/main", collections=collection)

skymap = "lsst_cells_v1"
skymapInfo = butler.get("skyMap", skymap=skymap, collections="skymaps")
tractInfo = skymapInfo.findTract(coord)
tract = tractInfo.tract_id
patchInfo = tractInfo.findPatch(coord)
patch = patchInfo.sequential_index
zooms_patch = zooms_patches.get(patch, {})

wcs = patchInfo.wcs
pixel = wcs.skyToPixel(coord)
coadd_x0, coadd_y0 = patchInfo.getOuterBBox().getBegin()
subpatch_rc = tuple(
    math.floor((pixel_ax - origin)/cutout_size_ax)
    for pixel_ax, cutout_size_ax, origin in zip(reversed(pixel), cutout_size, (coadd_y0, coadd_x0))
)
bbox_subpatch = Box2I(
    corner=Point2I(subpatch_rc[1]*cutout_size[0] + coadd_x0, subpatch_rc[0]*cutout_size[1] + coadd_y0),
    dimensions=cutout_size,
)
center_subpatch = patchInfo.wcs.pixelToSky(
    Point2D(
        (subpatch_rc[1] + 0.5)*cutout_size[0] + coadd_x0,
        (subpatch_rc[0] + 0.5)*cutout_size[1] + coadd_y0,
    )
)
center_subpatch_ap = SkyCoord(
    center_subpatch.getRa().asDegrees(), center_subpatch.getDec().asDegrees(), unit=u.deg,
)
zooms_subpatch = zooms_patch.get(subpatch_rc, tuple())

# This table has DESY6G quantities if needed too
# One can use the coord_best_[ra/dec] columns to plot all objects with the
# "best" available astrometry from a hierarchy of reference catalogs
# HST, DES, then ComCam
matched = butler.get(
    "matched_matched_cdfs_hlf_v2p1_euclid_q1_object",
    skymap=skymap, tract=tract, storageClass="ArrowAstropy", collections=collection,
)

kwargs_scatter_euclid = (
    dict(s=40, edgecolor="cornflowerblue", marker="D", facecolor="none", label="Euclid",),
    dict(s=60, edgecolor="orange", marker="D", facecolor="none", label="Euclid",),
)

parameters = {"bbox": bbox_subpatch}
cutouts = {
    band: butler.get("deep_coadd", skymap=skymap, tract=tract, patch=patch, band=band, parameters=parameters)
    for band in bands_lsst
}

n_cutouts = len(cutouts)

cutouts_lsst_rgb = tuple(
    np.sum([cutouts[band].image.array*bands_weights_lsst[band] for band in bands], axis=0)
    for bands in bands_lsst_rgb
)
img_rgb_lsst = lsstRGB(*cutouts_lsst_rgb, **kwargs_ppm)

weights_hst = 1/(u.ABmag.to(u.Jy, [abs_mag_sol_hst[band] for band in bands_hst]))
weights_hst_mean = np.mean(weights_hst)
# like with bands_weight_lsst, this re-weights per-band images so that
# solar colors appear white
bands_weights_hst = {
    band: weight_hst/weights_hst_mean for band, weight_hst in zip(bands_hst, weights_hst)
}

kwargs_ppm_hst = deepcopy(kwargs_ppm)
kwargs_ppm_hst["scaleLumKWargs"]["stretch"] = 80

cutouts_hst, extent = get_cutouts_cdfs_hst(
    tract, patch, bands_hst, skymap, center_subpatch_ap,
    cutout_size=tuple(int(math.ceil(s*scale_ratio_hst)) for s in cutout_size),
)

bands_euclid = ("vis",)

cutouts_euclid, extent_euclid = get_cutouts_euclid(
    skymap=skymap, tract=tract, patch=patch, bands=bands_euclid, position=center_subpatch_ap,
    cutout_size=tuple(2*s for s in cutout_size),
)
img_euclid = np.arcsinh(2e4*np.clip(cutouts_euclid["vis"].data, -1e-4, 0.01))

# Splitting off the plotting part for easy copypasting
matched_in = matched[
    (matched["coord_best_ra"] > extent[1]) & (matched["coord_best_ra"] < extent[0])
    & (matched["coord_best_dec"] > extent[2]) & (matched["coord_best_dec"] < extent[3])
]

# Make the finite (valid) euclid object ids positive
# This is critical to deal with the annoyance of masked id columns converting
# mask values to a negative (usually -1) fill value
matched_in["euclid_object_id"][matched_in["euclid_object_id"] < 0] *= -1

img_rgb_hst = lsstRGB(
    *(cutout.data * bands_weights_hst[band] for band, cutout in cutouts_hst.items()),
    **kwargs_ppm_hst,
)

# TODO compare extents?
fluxes_hst = {band: f"hst_f_{band.lower()}" for band in cutouts_hst.keys()}

zooms = (None,) + zooms_subpatch

for zoom in zooms:
    fig_hst, ax_hst = plot_external_matches(
        img_rgb_lsst, img_rgb_hst, extent, matched_in,
        id_ext="hst_id",
        bands_fluxes_ext=fluxes_hst,
    )

    ax_hst[0].set_title(f"HST {','.join(cutouts_hst.keys())}")
    title_lsst = f"{collection} LSST {';'.join(','.join(bands) for bands in bands_lsst_rgb)}"
    ax_hst[1].set_title(title_lsst)
    fig_hst.tight_layout()

    fig_euclid, ax_euclid = plot_external_matches(
        img_rgb_lsst, img_euclid, extent, matched_in,
        id_ext="euclid_object_id",
        bands_fluxes_ext={"vis": "euclid_flux_vis_sersic"},
        kwargs_imshow_ext={"cmap": "gray"},
    )

    ax_euclid[0].set_title(f"Euclid {','.join(bands_euclid)}")
    ax_euclid[1].set_title(title_lsst)
    fig_euclid.tight_layout()
    if zoom:
        for axes in (ax_euclid, ax_hst):
            for axis in axes:
                axis.set_xlim(*zoom[0])
                axis.set_ylim(*zoom[1])
                if zoom[0][0] > zoom[0][1]:
                    # I don't know why the y-axis needs inverting?
                    axis.invert_yaxis()
    plt.show()
