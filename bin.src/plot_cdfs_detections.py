# Load a patch of LSST data from the CDFS tract, then make diagnostic plots
# for detection and deblending. Compare to HST if there is overlap, and
# could show DES DECaLS detections if available (not implemented yet).

# See https://rubinobs.atlassian.net/browse/DM-48050 for more info

from collections import defaultdict
import math

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import astropy.units as u
import lsst.afw.image
from lsst.afw.table import SourceCatalog
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, pq
from lsst.geom import SpherePoint, degrees, Point2D, Extent2I
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({"image.origin": "lower", "font.size": 13, "figure.figsize": (18, 18)})

tract = 5063
patch = 5
plot_streaks_1 = False
if patch == 5:
    # A fairly dense region that ends up as one big blend at 100+ visits per band
    radec = 53.06700454971479, -28.221957930172174
elif patch == 34:
    plot_streaks_1 = False
    radec = 53.12277768, -27.73640709

weird_lines = []
if plot_streaks_1:
    # streaks, but at 90 deg? detector edges?
    radec = 53.0370158675743, -28.312212596429585
    weird_lines = [16532, 15098], [972, 651]
    weird_lines_slope = (weird_lines[1][1] - weird_lines[1][0]) / (
        weird_lines[0][1] - weird_lines[0][0]
    )
    weird_lines = [
        weird_lines,
        ([16238.5, 16238.5 - 100], [910, 910 + 100 / weird_lines_slope]),
        ([15712, 15512], [0, 200 / weird_lines_slope]),
    ]

# same
# radec = 52.92344351433806, -28.309346522676368
plot_footprint_band = "r"
cutout_asec = 100, 100

scale_lsst = 0.2
scale_hst = 0.03

kwargs_lup = dict(minimum=-0.5, Q=8, stretch=20)
kwargs_lup_hst = dict(minimum=-0.2, Q=8, stretch=2)

cen_cutout = SpherePoint(radec[0], radec[1], degrees)
extent_cutout = Extent2I(cutout_asec[0] / scale_lsst, cutout_asec[1] / scale_lsst)


def calibrate_exposure(exposure: lsst.afw.image.Exposure) -> lsst.afw.image.MaskedImageF:
    calib = exposure.getPhotoCalib()
    image = calib.calibrateImage(
        lsst.afw.image.MaskedImageF(exposure.image, mask=exposure.mask, variance=exposure.variance)
    )
    return image


def get_cutout_lsst(coadd, cen_cutout, extent_cutout):
    cutout = calibrate_exposure(coadd.getCutout(cen_cutout, extent_cutout))
    return cutout


def footprintsToNumpy(
    catalog: SourceCatalog,
    shape: tuple[int, int],
    xy0: tuple[int, int] | None = None,
    idx_offset: int = 0,
    negative_skipped: bool = True,
) -> np.ndarray:
    """Convert all of the footprints in a catalog into a boolean array.

    Parameters
    ----------
    catalog:
        The source catalog containing the footprints.
        This is typically a mergeDet catalog, or a full source catalog
        with the parents removed.
    shape:
        The final shape of the output array.
    xy0:
        The lower-left corner of the array that will contain the spans.

    Returns
    -------
    result:
        The array with pixels contained in `spans` marked as `True`.
    """
    if xy0 is None:
        offset = (0, 0)
    else:
        offset = (-xy0[0], -xy0[1])

    result = np.zeros(shape, dtype=int)
    for idx, src in enumerate(catalog[catalog["parent"] == 0]):
        spans = src.getFootprint().spans
        yidx, xidx = spans.shiftedBy(*offset).indices()
        idx_fp = src["id"] + idx_offset
        if negative_skipped and src["deblend_skipped"]:
            idx_fp = -idx_fp
        result[yidx, xidx] = idx_fp
    return result


def get_peak_xys(mergeDet):
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
    return x_peaks, y_peaks


def is_schema_item_deblend(schema_item):
    name = schema_item.field.getName()
    return "deblend" in name and ("error" not in name) and ("modelType" not in name)


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
    "u/dtaranu/DM-47234/20241101_20241113/match",
#    "LSSTComCam/runs/DRP/20241101_20241127/w_2024_48/DM-47841/match",
#    "LSSTComCam/runs/DRP/20241101_20241204/w_2024_49/DM-47988/match",
    "u/dtaranu/DM-47234/20241101_20241211/match",
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

figaxes_fp = {}
matches = {}

for collection in collections:
    matched = butler.get(
        "matched_cdfs_hlf_v2p1_objectTable_tract", tract=tract, skymap=name_skymap,
        collections=collection,
    )
    masked = matched["patch"].mask == True
    matched["patch"].mask = False
    patches_masked = [
        tractInfo.findPatch(
            SpherePoint(match["refcat_ra_gaia"], match["refcat_dec_gaia"], degrees)
        ).sequential_index
        for match in matched[masked]
    ]
    matched["patch"][masked] = patches_masked
    matched = matched[matched["patch"] == patch]
    n_matched = len(matched)
    masked = matched["objectId"].mask == True
    x, y = (np.zeros(n_matched, dtype=float) for _ in range(2))
    for column, value in (("coord_{c}", False), ("refcat_{c}_gaia", True)):
        rows = masked == value
        xy = np.array([
            [x for x in xy] for xy in tractInfo.wcs.skyToPixel([
                SpherePoint(row[column.format(c="ra")], row[column.format(c="dec")], degrees)
                for row in matched[rows]
            ])
        ])
        if len(xy) > 0:
            x[rows], y[rows] = xy[:, 0], xy[:, 1]
    matched["x"], matched["y"] = x, y
    matches[collection] = matched

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
        for collection, matched in matches.items():
            if "x_hst" not in matched.colnames:
                n_matched = len(matched)
                x, y = (np.zeros(n_matched, dtype=float) for _ in range(2))
                masked = matched["refcat_id"].mask == True
                for column, value in (("coord_{c}", True), ("refcat_{c}_gaia", False)):
                    rows = masked == value
                    xy = np.array([
                        [x for x in xy] for xy in (
                            wcs_cdfs.world_to_pixel(SkyCoord(
                                row[column.format(c="ra")]*u.degree, row[column.format(c="dec")]*u.degree,
                            ))
                            for row in matched[rows]
                        )
                    ])
                    x[rows], y[rows] = xy[:, 0], xy[:, 1]
                matched["x_hst"], matched["y_hst"] = x, y

        coadd_hst.data *= 10**(((u.nJy).to(u.ABmag) - img_cdfs[0].header["ZEROPNT"])/2.5)
        cutouts_hst[band_hst] = coadd_hst

    for collection in collections:
        for band in bands_lsst:
            if band not in cutouts_lsst:
                try:
                    coadd = butler.get(
                        "deepCoadd_calexp",
                        tract=tract, patch=patch, band=band, skymap=name_skymap, collections=collection,
                    )
                except dafButler.DatasetNotFoundError as e:
                    continue
                cutouts_lsst[collection][band] = get_cutout_lsst(coadd, cen_cutout, extent_cutout)
                if band == plot_footprint_band:
                    fig_fp, ax_fp = plt.subplots()
                    meas = butler.get(
                        "deepCoadd_meas",
                        tract=tract, patch=patch, band=band, skymap=name_skymap, collections=collection,
                    )
                    mergeDet = butler.get(
                        "deepCoadd_mergeDet", skymap=name_skymap, tract=tract, patch=patch,
                        collections=collection,
                    )
                    x_peaks, y_peaks = get_peak_xys(mergeDet)
                    x0, y0 = coadd.getBBox().getBegin()
                    img_fp = footprintsToNumpy(
                        meas, coadd.image.array.shape, xy0=(x0, y0),
                        idx_offset=2000-np.min(meas["id"]),
                    )
                    vmax = np.max(np.abs(img_fp))
                    ax_fp.imshow(
                        img_fp, cmap="gray", vmin=-vmax, vmax=vmax,
                        extent=[x0, x0 + img_fp.shape[1], y0, y0 + img_fp.shape[0]],
                    )
                    ax_fp.scatter(x_peaks, y_peaks, c='r', marker='+', label="peaks", s=50)
                    x, y = (meas[f"slot_Centroid_{c}"][meas["detect_isPrimary"]] for c in ("x", "y"))
                    ax_fp.scatter(x, y, edgecolor='royalblue', marker='o', label="primary", s=50, facecolor="None")
                    ax_fp.set_title(
                        f"{tract=} {patch=} footprints (dark=skipped, gray=none) {collection.split('/')[3]}"
                    )
                    ax_fp.legend()
                    fig_fp.tight_layout()
                    figaxes_fp[collection] = fig_fp, ax_fp


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

bands_rgb_hst = ("F775W", "F606W", "F435W")
if get_hst:
    img_rgb_hst = make_lupton_rgb(
        *[cutouts_hst[band].data*bands_weights_hst[band] for band in bands_rgb_hst],
        **kwargs_lup_hst
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
            try:
                coadd = butler.get(
                    "deepCoadd_calexp",
                    tract=tract, patch=patch, band=band, skymap=name_skymap, collections=collection,
                )
            except dafButler.DatasetNotFoundError as e:
                continue
            cutouts_lsst[collection][band] = get_cutout_lsst(coadd, cen_cutout, extent_cutout)


def plot_axis(
    axis, x, y, xy_peaks=None, xy_missing=None, title="", weird_lines=[],
    label="primary", label_missing="missing", **kwargs
):
    for x_wl, y_wl in weird_lines:
        axis.plot(x_wl, y_wl, 'r', zorder=0)
    if title:
        axis.set_title(title)
    axis.scatter(
        x, y,
        edgecolor='c', facecolor="None", marker='o', label=label, s=120, zorder=1,
    )
    if xy_peaks is not None:
        axis.scatter(
            xy_peaks[0], xy_peaks[1],
            c='c', marker='+', label="peaks", s=100, zorder=2,
        )
    if xy_missing is not None:
        axis.scatter(
            xy_missing[0], xy_missing[1],
            c='r', marker='x', label=label_missing, s=100, zorder=3,
        )
    axis.legend(**kwargs)


def is_within(x, y, extent):
    return (x > extent[0]) & (x < extent[1]) & (y > extent[2]) & (y < extent[3])


nrows, ncols = 1 + get_hst, len(collections)
fig_rgb, ax_rgb = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 12 * nrows))
fig_sn, ax_sn = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 12 * nrows))
fig_sigma, ax_sigma = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 12 * nrows))


for idx, (collection, cutouts_lsst_coll) in enumerate(cutouts_lsst.items()):
    matched = matches[collection]
    bbox = cutouts_lsst_coll["r"].getBBox()
    x_begin, y_begin = bbox.getBegin()
    x_end, y_end = bbox.getEnd()
    extent = [x_begin, x_end, y_begin, y_end]

    # Can change this to deepCoadd_det and add a band to show per-band peaks.
    # TODO: Figure out how to plot that coherently
    mergeDet = butler.get(
        "deepCoadd_mergeDet", skymap=name_skymap, tract=tract, patch=patch, collections=collection,
    )
    x_peaks, y_peaks = get_peak_xys(mergeDet)
    good_peaks = (x_peaks > x_begin) & (x_peaks < x_end) & (y_peaks > y_begin) & (y_peaks < y_end)
    xy_peaks = tuple(c[good_peaks] for c in (x_peaks, y_peaks))

    img_rgb_lsst = make_lupton_rgb(
        *[cutouts_lsst_coll[band].image.array * bands_weights_lsst[band] for band in bands_rgb_lsst],
        **kwargs_lup
    )
    name_short = collection.split("/")[3]
    axis = ax_rgb[idx][0] if get_hst else ax_rgb[idx]
    axis.imshow(img_rgb_lsst, extent=extent)

    good_lsst = matched["detect_isPrimary"] == True
    x_meas, y_meas = (matched[c][good_lsst] for c in ("x", "y"))
    good_meas = is_within(x_meas, y_meas, extent)
    missing = matched["objectId"].mask == True
    x_missing, y_missing = (matched[c][missing] for c in ("x", "y"))
    good_missing = is_within(x_missing, y_missing, extent)
    x_meas, y_meas = x_meas[good_meas], y_meas[good_meas]

    plot_axis(
        axis,
        x_meas, y_meas, xy_peaks, xy_missing=(x_missing[good_missing], y_missing[good_missing]),
        weird_lines=weird_lines, label_missing="HST unmatched",
        title=f"{tract=}, {patch=}, {','.join(bands_rgb_lsst)} {name_short}"
              f" n_primary={np.sum(good_meas)} n_peaks={len(xy_peaks[0])}",
        loc="upper left",
    )
    if get_hst:
        origin_hst = cutouts_hst[bands_rgb_hst[1]].origin_original
        shape_hst = img_rgb_hst.shape
        axis = ax_rgb[idx][1]
        extent_hst = [
            origin_hst[0], origin_hst[0] + shape_hst[1],
            origin_hst[1], origin_hst[1] + shape_hst[0],
        ]
        axis.imshow(img_rgb_hst, extent=extent_hst)
        good_hst = matched["refcat_id"].mask == False
        x_hst, y_hst = (matched[c][good_hst] for c in ("x_hst", "y_hst"))
        good_hst = is_within(x_hst, y_hst, extent_hst)
        missing = matched["refcat_id"].mask == True
        x_missing, y_missing = (matched[c][missing] for c in ("x_hst", "y_hst"))
        good_hst_miss = is_within(x_missing, y_missing, extent_hst)
        in_comcam = matched["objectId"].mask == False
        x_comcam, y_comcam = (matched[c][in_comcam] for c in ("x_hst", "y_hst"))
        good_comcam = is_within(x_comcam, y_comcam, extent_hst)
        plot_axis(
            axis,
            x_hst[good_hst], y_hst[good_hst], xy_peaks=(x_comcam[good_comcam], y_comcam[good_comcam]),
            xy_missing=(x_missing[good_hst_miss], y_missing[good_hst_miss]),
            weird_lines=weird_lines, label="HST obj", label_missing="ComCam unmatched",
            title=f"{tract=}, {patch=}, {','.join(bands_rgb_hst)} {name_short} n_hst={np.sum(good_hst)}",
        )

    bands_sn_coll = [band for band in bands_sn_lsst if band in cutouts_lsst_coll]
    img_sigma = np.sqrt(np.sum([cutouts_lsst_coll[band].variance.array for band in bands_sn_coll], axis=0))
    sigma_med = np.median(img_sigma)
    axis = ax_sigma[idx][0] if get_hst else ax_sigma[idx]
    axis.imshow(img_sigma, vmin=0.8*sigma_med, vmax=1.2*sigma_med, extent=extent, cmap="gray")
    plot_axis(
        axis,
        x_meas, y_meas, xy_peaks, weird_lines=weird_lines,
        title=f"{tract=}, {patch=}, {','.join(bands_sn_lsst)} sigma clip(0.8*median, 1.2*median) {name_short}"
              f" n_primary={np.sum(good_lsst)}",
    )
    img_sn = np.sum([cutouts_lsst_coll[band].image.array for band in bands_sn_coll], axis=0)/img_sigma
    axis = ax_sn[idx][0] if get_hst else ax_sn[idx]
    axis.imshow(np.clip(img_sn*(img_sn > 3), 0, 8), extent=extent, cmap="gray")
    plot_axis(
        axis,
        x_meas, y_meas, xy_peaks, weird_lines=weird_lines,
        title=f"{tract=}, {patch=}, {','.join(bands_sn_lsst)} S/N*(S/N >= 3) clip(0, 8) {name_short}"
              f" n_primary={np.sum(good_lsst)}",
    )


for fig in fig_rgb, fig_sn, fig_sigma:
    fig.tight_layout()

plt.show()
