import os.path

from astropy.visualization import make_lupton_rgb
from copy import deepcopy
import lsst.afw.image
import lsst.daf.butler as dafButler
import lsst.gauss2d as g2d
from lsst.geom import SpherePoint, degrees, Extent2I
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

get_hst = True
# This will use a lot of memory - each patch image is 4.1GB
keep_hst_full = True
plot_hst = True
use_ppm = True

if get_hst:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D, NoOverlapError
    from astropy.wcs import WCS
    import astropy.units as u
    import galsim as gs


if use_ppm:
    from lsst.pipe.tasks.prettyPictureMaker import lsstRGB

    kwargs_ppm = {
        "scaleLumKWargs": {"Q": 8.0, "stretch": 500, "highlight": 0.905882, "shadow": 0.12, "midtone": 0.25},
        "remapBoundsKwargs": {"absMax": 15000},
        "cieWhitePoint": (0.28, 0.28),
        "doLocalContrast": False,
        "scaleColorKWargs": {"maxChroma": 80, "saturation": 0.6},
    }


mpl.rcParams.update({"image.origin": "lower", 'font.size': 13})
kwargs_lup = dict(minimum=-2, Q=10, stretch=60)
kwargs_lup_hst = dict(minimum=0, Q=8, stretch=1.3)


def calibrate_exposure(exposure: lsst.afw.image.Exposure) -> lsst.afw.image.MaskedImageF:
    calib = exposure.getPhotoCalib()
    image = calib.calibrateImage(
        lsst.afw.image.MaskedImageF(exposure.image, mask=exposure.mask, variance=exposure.variance)
    )
    return image


# A nice group of galaxies
ra_gal, dec_gal = 53.12277768, -27.73640709
# ra_gal, dec_gal = 53.124654379926724, -27.740377354687737
# A region with some dubious detections
# ra_gal, dec_gal = 53.11, -27.9
# Another galaxy of some description
# ra_gal, dec_gal = 53.124848804610785, -27.758377013546585
# ra_gal, dec_gal = 52.9035204, -28.0243690  # nice galaxy but not fully covered
bands = ("i", "r", "g")
bands_hst = ("F775W", "F606W", "F435W")
weight_mean = np.mean([bands_weights_lsst[band] for band in bands])
for band in bands:
    bands_weights_lsst[band] /= weight_mean

scale_lsst = 0.2
scale_lsst_deg = 0.2/3600

coord = SpherePoint(ra_gal, dec_gal, degrees)
cutout_size = Extent2I(400, 400)
figsize_ax = 8

cutouts_hst = {}
wcs_hst = {}
full_hst = {}

butler = dafButler.Butler("/repo/main")
collections = [
    "u/dtaranu/DM-48367/v29_0_0_rc2/match",
]

n_collections = len(collections)
name_skymap = "lsst_cells_v1"
skymap = butler.get("skyMap", skymap=name_skymap, collections="skymaps")
tractInfo = skymap.findTract(coord)
tract = tractInfo.tract_id
patchInfo = tractInfo.findPatch(coord)
patch = patchInfo.sequential_index

# This table has DESY6G quantities if needed too
# One can use the coord_best_[ra/dec] columns to plot all objects with the
# "best" available astrometry from a hierarchy of reference catalogs
# HST, DES, then ComCam
matched = butler.get(
    "matched_matched_cdfs_hlf_v2p1_des_y6gold_object",
    skymap=name_skymap, tract=tract, storageClass="ArrowAstropy", collections=collections[0],
)

kwargs_scatter_lsst = (
    dict(s=40, edgecolor="lavender", marker="s", facecolor="none", label="LSST",),
    dict(s=60, edgecolor="aquamarine", marker="s", facecolor="none", label="LSST",),
)

kwargs_scatter_hst = (
    dict(s=40, edgecolor="cornflowerblue", marker="D", facecolor="none", label="HST",),
    dict(s=60, edgecolor="orange", marker="D", facecolor="none", label="HST",),
)


def scatter_multi(axis, x, y, kwargs_scatter_list):
    for kwargs_scatter in kwargs_scatter_list:
        axis.scatter(x, y, **kwargs_scatter)


coadds = {}
reference_coadds = {}

for collection in collections:
    try:
        coadds_collection = butler.query_datasets(
            "deep_coadd", collections=collection,
            skymap=name_skymap, tract=tract, patch=patch,
        )
    except dafButler.EmptyQueryResultError as err:
        continue

    coadds_iter = {
        coadd_ref.dataId["band"]: butler.get(coadd_ref) for coadd_ref in coadds_collection
    }
    # This fills in missing data in collections with incomplete coadds
    # (i.e. it should do nothing for runs with a fixed dataset, unless
    #  different visits ended up in the coadds)
    for band, coadd in coadds_iter.items():
        if band not in reference_coadds:
            reference_coadds[band] = coadd

    coadds_collection = {}
    # Reorder the coadds and fill in missing bands
    for band in bands:
        coadds_collection[band] = coadds_iter.get(band, reference_coadds[band])
    coadds[collection] = coadds_collection

cutouts = {
    collection: {
        band: calibrate_exposure(calexps.get(band).getCutout(coord, cutout_size)) for band in bands
    }
    for collection, calexps in coadds.items()
}

n_cutouts = len(cutouts)

if get_hst:
    abs_mag_hst = {
        "F435W": 5.35,
        "F606W": 4.72,
        "F775W": 4.52,
        "F814W": 4.52,
    }
    weights_hst = 1/(u.ABmag.to(u.Jy, [abs_mag_hst[band] for band in bands_hst]))
    weights_hst_mean = np.mean(weights_hst)
    # like with bands_weight_lsst, this re-weights per-band images so that
    # solar colors appear white
    bands_weights_hst = {
        band: weight_hst/weights_hst_mean for band, weight_hst in zip(bands_hst, weights_hst)
    }

    if use_ppm:
        kwargs_ppm_hst = deepcopy(kwargs_ppm)
        kwargs_ppm_hst["scaleLumKWargs"]["stretch"] = 80
        kwargs_ppm_hst["cieWhitePoint"] = (0.31, 0.31)

    path_cdfs_hst = f"/sdf/data/rubin/shared/hst/ecdfs/{name_skymap}"
    scale_hst = 0.03
    for band in bands_hst:
        if (fits_cdfs := full_hst.get(band)) is None:
            filename = (f"{path_cdfs_hst}/{tract}/{patch}/"
                        f"hlf_hst_coadd_{tract}_{patch}_{band}_{name_skymap}.fits.gz")
            if os.path.isfile(filename):
                fits_cdfs = fits.open(filename)
                if keep_hst_full:
                    full_hst[band] = fits_cdfs
                hdu_cdfs = fits_cdfs[1]
                wcs_cdfs = WCS(hdu_cdfs)
                try:
                    cutout = Cutout2D(
                        hdu_cdfs.data,
                        position=SkyCoord(ra_gal, dec_gal, unit=u.degree),
                        # Should read pixel scale from CD1_1, etc. oh well
                        size=(cutout_size[0]*scale_lsst/scale_hst, cutout_size[1]*scale_lsst/scale_hst),
                        wcs=wcs_cdfs,
                        copy=True,
                    )
                    cutouts_hst[band] = cutout
                except NoOverlapError as err:
                    continue

    if len(cutouts_hst) < 3:
        # We could try a little harder here, but never mind
        get_hst = False
        plot_hst = False
    else:
        cutout = next(iter(cutouts_hst.values()))
        radec_hst_begin = cutout.wcs.pixel_to_world(0, 0)
        radec_hst_end = cutout.wcs.pixel_to_world(cutout.shape[1], cutout.shape[0])
        (ra_begin, dec_begin), (ra_end, dec_end) = (
            (radec.ra.value, radec.dec.value) for radec in (radec_hst_begin, radec_hst_end)
        )
        extent_hst = (
            radec_hst_begin.ra.value, radec_hst_end.ra.value, radec_hst_begin.dec.value, radec_hst_end.dec.value,
        )
        img_lup_hst = make_lupton_rgb(
            *(cutout.data*bands_weights_hst[band] for band, cutout in cutouts_hst.items()),
            **kwargs_lup_hst,
        )

# Splitting off the plotting part for easy copypasting
if get_hst:
    extent = extent_hst
    matched_in = matched[
        (matched["coord_best_ra"] > extent[1]) & (matched["coord_best_ra"] < extent[0])
        & (matched["coord_best_dec"] > extent[2]) & (matched["coord_best_dec"] < extent[3])
    ]

    fig_hst, ax_hst = plt.subplots(figsize=(2*figsize_ax, 2*figsize_ax), nrows=1 + use_ppm)
    ax_plot = (ax_hst[0] if use_ppm else ax_hst)
    ax_plot.imshow(img_lup_hst, extent=extent_hst)

    good_hst = matched_in["matched_refcat_id"] >= 0
    scatter_multi(
        ax_plot,
        matched_in["matched_refcat_ra_gaia"][good_hst],
        matched_in["matched_refcat_dec_gaia"][good_hst],
        kwargs_scatter_list=kwargs_scatter_hst,
    )
    good_comcam = matched_in["objectId"] >= 0
    scatter_multi(
        ax_plot,
        matched_in["coord_ra"][good_comcam],
        matched_in["coord_dec"][good_comcam],
        kwargs_scatter_list=kwargs_scatter_lsst,
    )
    ax_plot.set_title(f"HST {','.join(cutouts_hst.keys())}")

    if use_ppm:
        img_ppm_hst = lsstRGB(
            *(cutout.data * bands_weights_hst[band] for band, cutout in cutouts_hst.items()),
            **kwargs_ppm_hst,
        )
        (ax_hst[1] if use_ppm else ax_hst).imshow(img_ppm_hst, extent=extent_hst)
    fig_hst.tight_layout()

n_cols = 1 + get_hst
n_rows = n_cutouts

figsize = (figsize_ax*n_cols, figsize_ax*n_rows)

fig_lup, ax_lup = plt.subplots(figsize=figsize, ncols=n_cols, nrows=n_rows)
if use_ppm:
    fig_ppm, ax_ppm = plt.subplots(figsize=figsize, ncols=n_cols, nrows=n_rows)

for idx, (collection, cutouts_bands) in enumerate(cutouts.items()):
    img_lup = make_lupton_rgb(
        *(cutout.image.array*bands_weights_lsst[band] for band, cutout in cutouts_bands.items()),
        **kwargs_lup,
    )
    if use_ppm:
        img_ppm = lsstRGB(
            *(cutout.image.array for band, cutout in cutouts_bands.items()),
            **kwargs_ppm,
        )
    title = f"{collection} LSST {','.join(cutouts_bands)}"
    if idx == 0:
        title = f"{title} {tract},{patch}"

    fig_ax_imgs = [
        (fig_lup, ax_lup, img_lup),
    ]
    if use_ppm:
        fig_ax_imgs.append((fig_ppm, ax_ppm, img_ppm))

    radec_begin = cutout.wcs.pixel_to_world(0, 0)
    radec_end = cutout.wcs.pixel_to_world(cutout.shape[1], cutout.shape[0])
    (ra_begin, dec_begin), (ra_end, dec_end) = (
        (radec.ra.value, radec.dec.value) for radec in (radec_begin, radec_end)
    )
    extent = (
        radec_begin.ra.value, radec_end.ra.value, radec_begin.dec.value, radec_end.dec.value,
    )

    for fig, axes, img in fig_ax_imgs:
        axis = (axes[idx, 0] if get_hst else axes[idx]) if (n_collections > 1) else axes[0]
        axis.imshow(img, extent=extent)
        axis.set_title(title)

        matched_in = matched[
            (matched["coord_best_ra"] > extent[1]) & (matched["coord_best_ra"] < extent[0])
            & (matched["coord_best_dec"] > extent[2]) & (matched["coord_best_dec"] < extent[3])
        ]
        good_comcam = matched_in["objectId"] >= 0
        ra, dec = (matched_in[f"coord_{c}"][good_comcam] for c in ("ra", "dec"))
        scatter_multi(axis, ra, dec, kwargs_scatter_list=kwargs_scatter_lsst)

        for idx_ell, (r_x, r_y, rho) in enumerate(zip(
            *(matched_in[f"sersic_{col}"][good_comcam] for col in ("reff_x", "reff_y", "rho"))
        )):
            ell_maj = g2d.EllipseMajor(g2d.Ellipse(r_x, r_y, rho), degrees=True)
            if ell_maj.r_major > 5:
                ell_patch = mpl.patches.Ellipse(
                    xy=(ra[idx_ell], dec[idx_ell]),
                    width=2*ell_maj.r_major*scale_lsst_deg,
                    height=2*ell_maj.r_major*ell_maj.axrat*scale_lsst_deg,
                    angle=-ell_maj.angle,
                    edgecolor='gray', fc='None', lw=2,
                )
                axis.add_artist(ell_patch)

        if get_hst:
            scatter_multi(
                axis,
                matched_in["matched_refcat_ra_gaia"][good_hst],
                matched_in["matched_refcat_dec_gaia"][good_hst],
                kwargs_scatter_list=kwargs_scatter_hst,
            )

    if get_hst:
        for fig, axes, img in fig_ax_imgs:
            # F775W is supposed be close to i-band, so we can do difference
            # imaging (sort of)
            band_diff_lsst, band_diff_hst = "i", "F775W"
            axis = (axes[idx, ] if get_hst else axes[idx]) if (n_collections > 1) else axes[1]
            coadd_lsst = coadds[collection][band_diff_lsst]
            cutout_lsst = cutouts[collection][band_diff_lsst]
            cutout_hst = cutouts_hst[band_diff_hst].data
            psf_r = coadd_lsst.psf.computeKernelImage(coadd_lsst.wcs.skyToPixel(coord))
            img_psf = gs.InterpolatedImage(gs.Image(psf_r.array, scale=0.2))
            cutout_hst_gs = gs.InterpolatedImage(gs.Image(cutout_hst, scale=scale_hst))
            # shift was eyeballed; should be estimated better
            # Would really be best to re-warp the HST coadd to the ComCam WCS
            # Also should convolve by the difference kernel but since the HST
            # PSF isn't readily available and it's so much smaller than the
            # ComCam PSF, ignore it
            img_convolved = gs.Convolve(cutout_hst_gs, img_psf).shift(0.1, 0.05).drawImage(
                nx=cutout_size[0], ny=cutout_size[1])
            # This hardcoded factor should be replaced with a real filter
            # transform derived from all stars in the whole field
            img_diff = (cutout_lsst.image.array - 1.095*img_convolved.array)
            max_diff = np.max(img_diff)
            axis.set_title("LSST - HST(x)LSST PSF diffim")
            axis.imshow(img_diff, cmap="gray", vmin=-max_diff, vmax=max_diff)

    for fig, *_ in fig_ax_imgs:
        fig.tight_layout()

plt.show()
