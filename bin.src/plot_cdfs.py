import lsst.afw.image
from astropy.visualization import make_lupton_rgb
from copy import deepcopy
import lsst.daf.butler as dafButler
from lsst.geom import SpherePoint, degrees, Extent2I
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

get_hst = True
plot_hst = True
use_ppm = True

if get_hst:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u
    import galsim as gs


if use_ppm:
    from lsst.pipe.tasks.prettyPictureMaker import lsstRGB
    kwargs_ppm = {
        "scaleLumKWargs": {"Q": 1.0, "stretch": 400, "max": 100},
        "remapBoundsKwargs": {"quant": 3.2},
        "cieWhitePoint": (0.31382, 0.3310),
        "doLocalContrast": False,
    }


mpl.rcParams.update({"image.origin": "lower", 'font.size': 13})


def calibrate_exposure(exposure: lsst.afw.image.Exposure) -> lsst.afw.image.MaskedImageF:
    calib = exposure.getPhotoCalib()
    image = calib.calibrateImage(
        lsst.afw.image.MaskedImageF(exposure.image, mask=exposure.mask, variance=exposure.variance)
    )
    return image


ra_gal, dec_gal = 53.124654379926724, -27.740377354687737
bands = ("i", "r", "g")
bands_hst = ("F814W", "F606W", "F435W")
weight_mean = np.mean([bands_weights_lsst[band] for band in bands])
for band in bands:
    bands_weights_lsst[band] /= weight_mean

scale_lsst = 0.2

coord = SpherePoint(ra_gal, dec_gal, degrees)
cutout_size = Extent2I(150, 150)
figsize_ax = 8
kwargs_lup = dict(minimum=-1, Q=8, stretch=100)

cutouts_hst = {}
wcs_hst = {}

if get_hst:
    abs_mag_hst = {
        "F435W": 5.35,
        "F606W": 4.72,
        "F814W": 4.52,
    }
    weights_hst = 1/(u.ABmag.to(u.Jy, [abs_mag_hst[band] for band in bands_hst]))
    weights_hst_mean = np.mean(weights_hst)
    bands_weights_hst = {
        band: weight_hst/weights_hst_mean for band, weight_hst in zip(bands_hst, weights_hst)
    }

    kwargs_lup_hst = deepcopy(kwargs_lup)
    kwargs_lup_hst["stretch"] = 4
    kwargs_lup_hst["minimum"] = -0.3

    if use_ppm:
        kwargs_ppm_hst = deepcopy(kwargs_ppm)
        kwargs_ppm_hst["scaleLumKWargs"]["stretch"] = 16

    path_cdfs_hst = "/sdf/data/rubin/user/dtaranu/tickets/cdfs/"
    scale_hst = 0.03
    for band in bands_hst:
        img_cdfs = fits.open(
            f"{path_cdfs_hst}hlsp_hlf_hst_acs-30mas_goodss_{band.lower()}_v2.0_sci.fits.gz"
        )
        wcs_cdfs = WCS(img_cdfs[0])
        cutout = Cutout2D(
            img_cdfs[0].data,
            position=SkyCoord(ra_gal, dec_gal, unit=u.degree),
            # Should read pixel scale from CD1_1, etc. oh well
            size=(cutout_size[0]*scale_lsst/scale_hst, cutout_size[1]*scale_lsst/scale_hst),
            wcs=wcs_cdfs,
            copy=True,
        )
        cutout.data *= 10**(((u.nJy).to(u.ABmag) - img_cdfs[0].header["ZEROPNT"])/2.5)
        cutouts_hst[band] = cutout

    img_lup_hst = make_lupton_rgb(
        *(cutout.data*bands_weights_hst[band] for band, cutout in cutouts_hst.items()),
        **kwargs_lup_hst,
    )

    extent_hst = (0, img_lup_hst.shape[1]*scale_hst, 0, img_lup_hst.shape[0]*scale_hst)

    fig_hst, ax_hst = plt.subplots(figsize=(2*figsize_ax, 2*figsize_ax), nrows=1 + use_ppm)
    (ax_hst[0] if use_ppm else ax_hst).imshow(img_lup_hst, extent=extent_hst)

    if use_ppm:
        img_ppm = lsstRGB(
            *(cutout.data for band, cutout in cutouts_hst.items()),
            **kwargs_ppm_hst,
        )
        (ax_hst[1] if use_ppm else ax_hst).imshow(img_lup_hst, extent=extent_hst)

butler = dafButler.Butler("/repo/embargo")
collections = butler.registry.queryCollections(
    "LSSTComCam/runs/nightlyValidation/202411*",
    collectionTypes=[dafButler.CollectionType.CHAINED],
)
name_skymap = "lsst_cells_v1"
skymap = butler.get("skyMap", skymap=name_skymap, collections="skymaps")
tractInfo = skymap.findTract(coord)
tract = tractInfo.tract_id
patchInfo = tractInfo.findPatch(coord)
patch = patchInfo.sequential_index

coadds = {}
reference_coadds = {}

for collection in collections:
    try:
        coadds_collection = butler.query_datasets(
            "deepCoadd_calexp", collections=collection,
            skymap=name_skymap, tract=tract, patch=patch,
        )
    except dafButler.EmptyQueryResultError as err:
        continue

    coadds_iter = {
        coadd_ref.dataId["band"]: butler.get(coadd_ref) for coadd_ref in coadds_collection
    }
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
    title = collection
    if idx == 0:
        title = f"{title} {ra_gal:.7f}, {dec_gal:.7f}"

    fig_ax_imgs = [
        (fig_lup, ax_lup, img_lup),
    ]
    if use_ppm:
        fig_ax_imgs.append((fig_ppm, ax_ppm, img_ppm))

    extent = (0., cutout_size[0] * 0.2, 0., cutout_size[1] * 0.2)

    for fig, axes, img in fig_ax_imgs:
        axis = axes[idx, 0] if get_hst else axes[idx]
        axis.imshow(img, extent=extent)
        axis.set_title(title)

    if get_hst:
        for fig, axes, img in fig_ax_imgs:
            band_diff_lsst, band_diff_hst = "i", "F814W"
            axis = axes[idx, 1] if get_hst else axes[idx]
            coadd_lsst = coadds[collection][band_diff_lsst]
            cutout_lsst = cutouts[collection][band_diff_lsst]
            cutout_hst = cutouts_hst[band_diff_hst].data
            psf_r = coadd_lsst.psf.computeKernelImage(coadd_lsst.wcs.skyToPixel(coord))
            img_psf = gs.InterpolatedImage(gs.Image(psf_r.array, scale=0.2))
            cutout_hst_gs = gs.InterpolatedImage(gs.Image(cutout_hst, scale=scale_hst))
            # shift was eyeballed; should be estimated better
            img_convolved = gs.Convolve(cutout_hst_gs, img_psf).shift(0.1, 0.05).drawImage(
                nx=cutout_size[0], ny=cutout_size[1])
            img_diff = (cutout_lsst.image.array - 1.095*img_convolved.array)
            max_diff = np.max(img_diff)
            axis.imshow(img_diff, cmap="gray", vmin=-max_diff, vmax=max_diff)

    for fig, *_ in fig_ax_imgs:
        fig.tight_layout()

plt.show()
