# Load a patch of LSST data from the CDFS tract, then make diagnostic plots
# for detection and deblending. Compare to HST if there is overlap, and
# show DES DECaLS detections if available.

from collections import defaultdict
import math

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import astropy.units as u
import lsst.afw.image
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, pq
from lsst.geom import SpherePoint, degrees, Point2D, Extent2I
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({"image.origin": "lower", "font.size": 13, "figure.figsize": (18, 18)})


def calibrate_exposure(exposure: lsst.afw.image.Exposure) -> lsst.afw.image.MaskedImageF:
    calib = exposure.getPhotoCalib()
    image = calib.calibrateImage(
        lsst.afw.image.MaskedImageF(exposure.image, mask=exposure.mask, variance=exposure.variance)
    )
    return image


tract = 5063
patch = 5
radec = 53.06700454971479, -28.221957930172174
cutout_asec = 120, 120

scale_lsst = 0.2
scale_hst = 0.03

cen_cutout = SpherePoint(radec[0], radec[1], degrees)
extent_cutout = Extent2I(cutout_asec[0] / scale_lsst, cutout_asec[1] / scale_lsst)

# There's a little bit of data in 26, 36, 46, but not much
# 24, 25, 34, 35 are the only ones with > 90% coverage in at least one band
# 34 appears to have full coverage in at least one band
get_hst = patch in (
    23, 24, 25,
    33, 34, 35,
    34, 44, 45,
)

name_skymap = "lsst_cells_v1"
butler = dafButler.Butler("/repo/embargo")
skymap = butler.get("skyMap", skymap=name_skymap, collections="skymaps")
tractInfo = skymap[tract]
collections = (
    "LSSTComCam/runs/DRP/20241101_20241127/w_2024_48/DM-47841",
    "LSSTComCam/runs/DRP/20241101_20241204/w_2024_49/DM-47988",
)

bands_hst_lsst = {
    "F435W": ("g",),
    "F606W": ("r",),
    "F775W": ("i",),
    "F814W": ("i", "z"),
}

path_cdfs_hst = "/sdf/data/rubin/user/dtaranu/tickets/cdfs/"

patchInfo = tractInfo[patch]
bbox_outer = patchInfo.outer_bbox

(ra_begin, dec_begin), (ra_end, dec_end) = (
    (radec[0] + cutout_asec[0]/2.0, radec[0] - cutout_asec[0]/2.0),
    (radec[1] - cutout_asec[1]/2.0, radec[1] + cutout_asec[1]/2.0),
)

cutouts_hst = {}
cutouts_lsst = defaultdict(dict)


def get_cutout_lsst(butler, cen_cutout, extent_cutout, **kwargs):
    coadd = butler.get("deepCoadd_calexp", **kwargs)
    cutout = calibrate_exposure(coadd.getCutout(cen_cutout, extent_cutout))
    return cutout


for band_hst, bands_lsst in bands_hst_lsst.items():
    if get_hst:
        img_cdfs = fits.open(
            f"{path_cdfs_hst}hlsp_hlf_hst_acs-30mas_goodss_{band_hst.lower()}_v2.0_sci.fits.gz"
        )
        wcs_cdfs = WCS(img_cdfs[0])

        coadd_hst = Cutout2D(
            img_cdfs[0].data,
            position=SkyCoord(radec[0]*u.degree, radec[1]*u.degree),
            size=(
                int(math.ceil(cutout_asec[0]/scale_hst)),
                int(math.ceil(cutout_asec[1]/scale_hst)),
            ),
            wcs=wcs_cdfs,
            copy=True,
        )
        coadd_hst.data *= 10**(((u.nJy).to(u.ABmag) - img_cdfs[0].header["ZEROPNT"])/2.5)
        cutouts_hst[band_hst] = coadd_hst

    for collection in collections:
        for band in bands_lsst:
            if band not in cutouts_lsst:
                cutouts_lsst[collection][band] = get_cutout_lsst(
                    butler, cen_cutout, extent_cutout,
                    tract=tract, patch=patch, band=band, skymap=name_skymap, collections=collection,
                )

weight_mean = np.mean([bands_weights_lsst[band] for band in cutouts_lsst[collections[0]]])
for band in cutouts_lsst[collections[0]]:
    bands_weights_lsst[band] /= weight_mean
abs_mag_hst = {
    "F435W": 5.35,
    "F606W": 4.72,
    "F775W": 4.52,
    "F814W": 4.52,
}
weights_hst = 1/(u.ABmag.to(u.Jy, [abs_mag_hst[band] for band in cutouts_hst]))
weights_hst_mean = np.mean(weights_hst)
bands_weights_hst = {
    band: weight_hst/weights_hst_mean for band, weight_hst in zip(cutouts_hst, weights_hst)
}
kwargs_lup = dict(minimum=-1, Q=8, stretch=16)
kwargs_lup_hst = dict(minimum=-0.3, Q=8, stretch=4)

if get_hst:
    img_rgb_hst = make_lupton_rgb(
        *[cutouts_hst[band].image.array*bands_weights_lsst[band]
          for band in ("F775W", "F606W", "F435W")],
        **kwargs_lup
    )

load_decals = False
if load_decals:
    cat_decals = arrow_to_astropy(pq.read_table(
        "/sdf/data/rubin/user/dtaranu/tickets/DM-47234/decals_dr10_lsst_cells_v1_5063.parq"
    ))

bands_rgb_lsst = ("i", "r", "g")
bands_sn_lsst = ("u", "g", "r", "i", "z", "y")
for band in bands_sn_lsst:
    if band not in cutouts_lsst:
        for collection in collections:
            cutouts_lsst[collection][band] = get_cutout_lsst(
                butler, cen_cutout, extent_cutout,
                tract=tract, patch=patch, band=band, skymap=name_skymap, collections=collection,
            )


def is_schema_item_deblend(schema_item):
    name = schema_item.field.getName()
    return "deblend" in name and ("error" not in name) and ("modelType" not in name)


nrows, ncols = 1 + get_hst, len(collections)
fig_rgb, ax_rgb = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 12 * nrows))
fig_sn, ax_sn = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 12 * nrows))


for idx, (collection, cutouts_lsst_coll) in enumerate(cutouts_lsst.items()):
    objects = butler.get(
        "objectTable_tract", skymap=name_skymap, tract=tract, storageClass="ArrowAstropy",
        collections=collection,
        parameters={"columns": ("objectId", "patch", "coord_ra", "coord_dec", "x", "y", "detect_isPrimary")}
    )
    objects_patch = objects[objects["patch"] == patch]
    x, y = (objects_patch[c] for c in ("x", "y"))
    bbox = cutouts_lsst_coll["r"].getBBox()
    x_begin, y_begin = bbox.getBegin()
    x_end, y_end = bbox.getEnd()
    extent = [x_begin, x_end, y_begin, y_end]
    good = objects_patch["detect_isPrimary"] & (x > x_begin) & (x < x_end) & (y > y_begin) & (y < y_end)
    img_rgb_lsst = make_lupton_rgb(
        *[cutouts_lsst_coll[band].image.array * bands_weights_lsst[band] for band in bands_rgb_lsst],
        **kwargs_lup
    )
    name_short = collection.split("/")[3]
    axis = ax_rgb[idx][0] if get_hst else ax_rgb[idx]
    axis.imshow(img_rgb_lsst, extent=extent)
    axis.scatter(x[good], y[good], c='c', marker='+', label="primary", s=120)
    axis.set_title(f"{tract=}, {patch=}, {','.join(bands_rgb_lsst)} {name_short} n_primary={np.sum(good)}")
    if get_hst:
        ax_rgb[idx][1].imshow(cutouts_hst[band])

    mergeDet = butler.get(
        "deepCoadd_mergeDet", skymap=name_skymap, tract=tract, patch=patch, collections=collection,
    )
    n_peaks = np.sum([len(det.getFootprint().getPeaks()) for det in mergeDet])
    idx_peak = 0
    x_peaks = np.empty(n_peaks, dtype=int)
    y_peaks = np.empty(n_peaks, dtype=int)
    for det in mergeDet:
        peaks = det.getFootprint().getPeaks()
        n_peaks = len(peaks)
        x_peaks[idx_peak:idx_peak + n_peaks] = peaks["i_x"]
        y_peaks[idx_peak:idx_peak + n_peaks] = peaks["i_y"]
        idx_peak += n_peaks
    good_peaks = (x_peaks > x_begin) & (x_peaks < x_end) & (y_peaks > y_begin) & (y_peaks < y_end)
    axis.scatter(
        x_peaks[good_peaks], y_peaks[good_peaks],
        edgecolor='c', marker='o', label="peaks", s=100, facecolor="None",
    )
    axis.legend()

    meas = butler.get(
        "deepCoadd_meas", tract=tract, patch=patch, band=band, skymap=name_skymap,
        collections=collection,
    )
    x_meas, y_meas = (meas[f"slot_Centroid_{c}"] for c in ("x", "y"))
    good_meas = (x_meas > x_begin) & (x_meas < x_end) & (y_meas > y_begin) & (y_meas < y_end)

    print(f"{collection=} n_within={np.sum(good_meas)}\ncolumn, min, max, sum\n")
    cols_blend = [x.field.getName() for x in meas.getSchema() if is_schema_item_deblend(x)]
    for col in cols_blend:
        print(
            col,
            np.nanmin(meas[good_meas][col]), np.nanmax(meas[good_meas][col]),
            np.nansum(meas[good_meas][col]),
        )

    img_sn = np.sum([cutouts_lsst_coll[band].image.array for band in bands_sn_lsst], axis=0)
    img_sn /= np.sqrt(np.sum([cutouts_lsst_coll[band].variance.array for band in bands_sn_lsst], axis=0))
    axis = ax_sn[idx][0] if get_hst else ax_sn[idx]
    axis.imshow(np.clip(img_sn*(img_sn > 3), 0, 8), extent=extent, cmap="gray")
    axis.set_title(
        f"{tract=}, {patch=}, {','.join(bands_sn_lsst)} S/N*(S/N >= 3) clip(0, 8) {name_short}"
        f" n_primary={np.sum(good)}"
    )
    axis.scatter(x[good], y[good], c='c', marker='+', label="primary", s=120)
    axis.scatter(
        x_peaks[good_peaks], y_peaks[good_peaks],
        edgecolor='c', marker='o', label="peaks", s=100, facecolor="None",
    )
    axis.legend()

for fig in fig_rgb, fig_sn:
    fig.tight_layout()

plt.show()
