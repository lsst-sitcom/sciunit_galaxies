# Load a patch of LSST data from the CDFS tract, then make diagnostic plots
# for detection and deblending.

# This is specifically looking for collinear detections which have been
# diagnosed as a symptom of unmasked defects.
# See https://rubinobs.atlassian.net/browse/DM-48174 for details.

from collections import defaultdict

from astropy.visualization import make_lupton_rgb
import lsst.afw.image
import lsst.daf.butler as dafButler
from lsst.geom import degrees, Box2I, Extent2I, Point2I, SpherePoint
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({"image.origin": "lower", "font.size": 13, "figure.figsize": (18, 18)})

tract = 5063
patch = 5
radec = 53.0298882882399, -28.30743291598709

weird_lines = [16532, 15098], [972, 651]
weird_lines_slope = (weird_lines[1][1] - weird_lines[1][0]) / (
    weird_lines[0][1] - weird_lines[0][0]
)
weird_lines = [
    weird_lines,
    ([16238.5, 16238.5 - 100], [910, 910 + 100 / weird_lines_slope]),
    ([15712, 15512], [0, 200 / weird_lines_slope]),
]

cutout_asec = 180, 180

cen_cutout = SpherePoint(radec[0], radec[1], degrees)


def calibrate_exposure(exposure: lsst.afw.image.Exposure) -> lsst.afw.image.MaskedImageF:
    calib = exposure.getPhotoCalib()
    image = calib.calibrateImage(
        lsst.afw.image.MaskedImageF(exposure.image, mask=exposure.mask, variance=exposure.variance)
    )
    return image


def get_cutout_lsst(coadd, cen_cutout, extent_cutout):
    cutout = calibrate_exposure(coadd.getCutout(cen_cutout, extent_cutout))
    return cutout


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


def plot_axis(axis, x, y, good, x_peaks, y_peaks, good_peaks, title="", weird_lines=[]):
    for x_wl, y_wl in weird_lines:
        axis.plot(x_wl, y_wl, 'r', zorder=0)
    if title:
        axis.set_title(title)
    axis.scatter(
        x[good], y[good],
        edgecolor='c', facecolor="None", marker='o', label="primary", s=120, zorder=1,
    )
    axis.scatter(
        x_peaks[good_peaks], y_peaks[good_peaks],
        c='c', marker='+', label="peaks", s=100, zorder=2,
    )
    axis.legend()

def get_xy_corners(ra_begin, ra_end, dec_begin, dec_end, wcs):
    return (
        (int(c) for c in wcs.skyToPixel(SpherePoint(ra, dec, degrees)))
        for ra, dec in ((ra_begin, dec_begin), (ra_end, dec_end))
    )

name_skymap = "lsst_cells_v1"
butler = dafButler.Butler("/repo/embargo")
skymap = butler.get("skyMap", skymap=name_skymap, collections="skymaps")
tractInfo = skymap[tract]
collections = (
    "LSSTComCam/runs/DRP/20241101_20241120/w_2024_47/DM-47746",
    "LSSTComCam/runs/DRP/20241101_20241127/w_2024_48/DM-47841",
#    "LSSTComCam/runs/DRP/20241101_20241204/w_2024_49/DM-47988",
)

patchInfo = tractInfo[patch]
bbox_outer = patchInfo.outer_bbox

cosdec = np.cos(radec[1]*np.pi/180.)

(ra_begin, ra_end), (dec_begin, dec_end) = (
    (radec[0] + cutout_asec[0]/(cosdec*7200.0), radec[0] - cutout_asec[0]/(cosdec*7200.0)),
    (radec[1] - cutout_asec[1]/7200.0, radec[1] + cutout_asec[1]/7200.0),
)
(x_begin, y_begin), (x_end, y_end) = get_xy_corners(ra_begin, ra_end, dec_begin, dec_end, tractInfo.wcs)
bbox = Box2I(Point2I(x_begin, y_begin), Extent2I(x_end - x_begin, y_end - y_begin))

cutouts_hst = {}
cutouts_lsst = defaultdict(dict)

figaxes_fp = {}

bands_lsst = ("u", "g", "r", "i", "z", "y")
bands_rgb = ("i", "r", "g")
weight_mean = np.mean([bands_weights_lsst[band] for band in bands_rgb])
for band in bands_rgb:
    bands_weights_lsst[band] /= weight_mean
kwargs_lup = dict(minimum=-0.1, Q=8, stretch=2)

cutouts_collection = collections[0]
cutouts_visits = None

fig_rgb, ax_rgb = plt.subplots(nrows=1, ncols=len(collections), figsize=(12*len(collections), 12))

for idx, collection in enumerate(collections):
    objects = butler.get(
        "objectTable_tract", skymap=name_skymap, tract=tract, storageClass="ArrowAstropy",
        collections=collection,
        parameters={"columns": ("objectId", "patch", "coord_ra", "coord_dec", "x", "y", "detect_isPrimary")}
    )
    objects_patch = objects[objects["patch"] == patch]
    x, y = (objects_patch[c] for c in ("x", "y"))

    cutouts = {}
    for band in bands_lsst:
        cutouts[band] = butler.get(
            "deepCoadd_calexp",
            tract=tract, patch=patch, band=band, skymap=name_skymap, collections=collection,
            parameters={"bbox": bbox},
        )
    if collection == cutouts_collection:
        cutouts_visits = cutouts

    bbox = cutouts[bands_rgb[1]].getBBox()
    x_begin, y_begin = bbox.getBegin()
    x_end, y_end = bbox.getEnd()
    extent = [x_begin, x_end, y_begin, y_end]
    good = objects_patch["detect_isPrimary"] & (x > x_begin) & (x < x_end) & (y > y_begin) & (y < y_end)

    mergeDet = butler.get(
        "deepCoadd_mergeDet", skymap=name_skymap, tract=tract, patch=patch, collections=collection,
    )
    x_peaks, y_peaks = get_peak_xys(mergeDet)
    good_peaks = (x_peaks > x_begin) & (x_peaks < x_end) & (y_peaks > y_begin) & (y_peaks < y_end)

    img_rgb_lsst = make_lupton_rgb(
        *[cutouts[band].image.array * bands_weights_lsst[band] for band in bands_rgb],
        **kwargs_lup
    )
    name_short = collection.split("/")[3]
    axis = ax_rgb[idx]
    axis.imshow(img_rgb_lsst, extent=extent)
    axis.autoscale(enable=False)
    plot_axis(
        axis,
        x, y, good, x_peaks, y_peaks, good_peaks, weird_lines=weird_lines,
        title=f"{tract=}, {patch=}, {','.join(bands_rgb)} {name_short} n_primary={np.sum(good)}",
    )

    nrows, ncols = 3, 2
    fig_sig, ax_sig = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 24))

    row, col = 0, 0
    for band, cutout in cutouts.items():
        img_sigma = np.sqrt(cutout.variance.array)
        sigma_med = np.median(img_sigma)
        axis = ax_sig[row, col]
        axis.imshow(img_sigma, vmin=0.8*sigma_med, vmax=1.2*sigma_med, extent=extent, cmap="gray")
        axis.autoscale(enable=False)
        plot_axis(
            axis,
            x, y, good, x_peaks, y_peaks, good_peaks, weird_lines=weird_lines,
            title=f"{tract=}, {patch=}, {band=}, sigma clip(0.8*median, 1.2*median) {name_short}",
        )
        col += 1
        if col == ncols:
            row += 1
            col = 0
    fig_sig.tight_layout()

for fig in (fig_rgb,):
    fig.tight_layout()

collection = cutouts_collection
ccds = cutouts_visits["i"].getInfo().getCoaddInputs().ccds

row, col = 0, 0
nrows, ncols = 2, 2
fig_ccd, ax_ccd = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))

for ccd in ccds:
    calexp_wcs = butler.get(
        "calexp.wcs", visit=ccd["visit"], instrument="LSSTComCam",
        detector=ccd["ccd"], collections=collection,
    )
    (x_bc, y_bc), (x_ec, y_ec) = get_xy_corners(ra_begin, ra_end, dec_begin, dec_end, calexp_wcs)
    if x_bc > x_ec:
        x_bc, x_ec = x_ec, x_bc
    if y_bc > y_ec:
        y_bc, y_ec = y_ec, y_bc
    if (x_ec < ccd["bbox_min_x"]) or (x_bc > ccd["bbox_max_x"]) or (
            y_ec < ccd["bbox_min_y"]) or (y_bc > ccd["bbox_max_y"]):
        continue
    x_bc, y_bc = (max(v, ccd[f"bbox_min_{c}"]) for v, c in ((x_bc, "x"), (y_bc, "y")))
    x_ec, y_ec = (min(v, ccd[f"bbox_max_{c}"]) for v, c in ((x_ec, "x"), (y_ec, "y")))
    cutout = butler.get(
        "calexp",
        visit=ccd["visit"], instrument="LSSTComCam", detector=ccd["ccd"], collections=collection,
        parameters={"bbox": Box2I(Point2I(x_bc, y_bc), Extent2I(x_ec - x_bc, y_ec - y_bc))},
    )
    x_pc, y_pc = [], []
    for _x, _y in zip(x_peaks, y_peaks):
        x_c, y_c = calexp_wcs.skyToPixel(tractInfo.wcs.pixelToSky(_x, _y))
        if (x_c >= x_bc) and (x_c <= x_ec) and (y_c >= y_bc) and (y_c <= y_ec):
            x_pc.append(x_c)
            y_pc.append(y_c)

    axis_img = ax_ccd[row, 0]
    axis_sn = ax_ccd[row, 1]
    axis_img.imshow(np.arcsinh(cutout.image.array), extent=[x_bc, x_ec, y_bc, y_ec], cmap="gray")
    img_sn = cutout.image.array/np.sqrt(cutout.variance.array)
    axis_sn.imshow((img_sn >= 3)*np.clip(img_sn, 0, 8), extent=[x_bc, x_ec, y_bc, y_ec], cmap="gray")
    parallel = True
    for idx_wl, (x_wl, y_wl) in enumerate(weird_lines):
        x_wlc, y_wlc = [], []
        for _x, _y in zip(x_wl, y_wl):
            x_c, y_c = calexp_wcs.skyToPixel(tractInfo.wcs.pixelToSky(_x, _y))
            x_wlc.append(x_c)
            y_wlc.append(y_c)
        if idx_wl == 0:
            slope = (y_wlc[1] - y_wlc[0])/(x_wlc[1] - x_wlc[0])
            if not ((np.abs(slope) < 0.05) or (np.abs(slope) > 20)):
                parallel = False
                break
        for axis in axis_img, axis_sn:
            axis.autoscale(enable=False)
            axis.plot(x_wlc, y_wlc, 'b', zorder=0)
    if not parallel:
        continue
    for axis in axis_img, axis_sn:
        axis.scatter(x_pc, y_pc, c='orange', marker='+', label="peaks", s=100, zorder=2)
    axis_img.set_title(f"visit={ccd['visit']} detector={ccd['ccd']} ({ccd['filter'][0]}) {name_short}")
    axis_sn.set_title("S/N clip (>3, <8)")
    row += 1
    if row == nrows:
        row, col = 0, 0
        fig_ccd.tight_layout()
        plt.show()
        plt.close(fig_ccd)
        del ax_ccd
        del fig_ccd
        fig_ccd, ax_ccd = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
