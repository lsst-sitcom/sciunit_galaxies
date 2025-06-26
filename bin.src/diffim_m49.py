# Take an HST image of M49, reproject with the ComCam WCS, convolve with
# the ComCam PSF and make a difference image

from astropy.io import fits
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import galsim as gs
import lsst.daf.butler as dafButler
from lsst.geom import Box2I, Extent2D, Point2I, SpherePoint, degrees
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from reproject import reproject_interp
from scipy.signal import fftconvolve

use_color_hst = True

mpl.rcParams.update({"image.origin": "lower", "font.size": 13, "figure.figsize": (18, 18)})

path_hst = "/sdf/data/rubin/user/dtaranu/tickets/virgo"
collections = (
    "u/jbosch/DM-50735/a",
    "u/lskelvin/scratch/DM-50782-v2",
)
butler = dafButler.Butler("/repo/embargo")
tract = 10804
patch = 35

coord = SpherePoint(187.444900, 8.00040, degrees)

xmin, ymin = 150, 120
xmax, ymax = 8760, 8760
if use_color_hst:
    hdu_hst = fits.open(f"{path_hst}/color_hst_mos_1040600_acs_wfc_f850lp_long_f814w_f475w_sci.fits")
    img_hst_ri = hdu_hst[0].data[1, :, :]
    header_hst_img = hdu_hst[0].header
    # these are sufficient for F850LP/F475W
    # xmin, ymin = 1425, 700
    xmax, ymax = 14650, 13075
    mask_hst_ri = ~np.isfinite(img_hst_ri[ymin:ymax, xmin:xmax])
    bands = ("g", "r", "i", "z")
    zp_ab = 25.954
else:
    hdu_hst = fits.open(f"{path_hst}/hst_mos_1040600_acs_wfc_f814w_drz.fits")
    img_hst_ri = hdu_hst[1].data
    mask_hst_ri = ~(hdu_hst[2].data[ymin:ymax, xmin:xmax] > 0)
    header_hst_img = hdu_hst[1].header
    bands = ("r", "i")
    zp_ab = -2.5*np.log10(header_hst_img["PHOTFLAM"]) - 5*np.log10(header_hst_img["PHOTPLAM"]) - 2.408
    del hdu_hst

wcs_hst = WCS(header_hst_img)
if use_color_hst:
    wcs_hst = wcs_hst.sub(2)

cutout_hst_ri = Cutout2D(
    img_hst_ri,
    position=((xmin + xmax)/2., (ymin + ymax)/2.),
    size=(ymax - ymin, xmax - xmin),
    wcs=wcs_hst,
)
cutout_hst_ri.data[mask_hst_ri] = 0
# Guesstimated mode
cutout_hst_ri.data[~mask_hst_ri] -= 0.12
cutout_hst_ri.data[~mask_hst_ri] *= (u.ABmag*zp_ab).to(u.nJy).value
wcs_hst = cutout_hst_ri.wcs
cutouts_hst = {"ri": cutout_hst_ri.data}
masks_hst = {"ri": mask_hst_ri}

if use_color_hst:
    img_hst_z, img_hst_g = (hdu_hst[0].data[idx, ymin:ymax, xmin:xmax] for idx in (0, 2))
    mask_hst = ~np.isfinite(img_hst_g) | ~np.isfinite(img_hst_z)

    # The modes are guesstimates from the histogram
    for band, img_hst, mask, zp_ab, mode_img in (
        ("g", img_hst_g, mask_hst, 26.071, 0.0414),
        ("z", img_hst_z, mask_hst, 24.872, 0.00286),
    ):
        img_hst[mask] = 0
        img_hst[~mask] -= mode_img
        img_hst[~mask] *= (u.ABmag*zp_ab).to(u.nJy).value
        cutouts_hst[band] = img_hst
        masks_hst[band] = mask

    del hdu_hst

coadd_x0, coadd_y0 = skymap_obj[tract][patch].getOuterBBox().getBegin()
bbox = Box2I(
    minimum=Point2I(0 + coadd_x0, 1500 + coadd_y0),
    maximum=Point2I(3400 + coadd_x0 - 1, 3400 + coadd_y0 - 1),
)

do_cache = True
cache = {}
if do_cache:
    cutouts_hst_convolved = {}
    ratios = {}

for collection in collections[:1]:
    coadds = {}
    coadd_wcs_shifted = None
    xy0 = None
    psfs_hst = {}
    cache_collection = cache.get(collection, {})
    for band in bands:
        coadd = butler.get(
            "deep_coadd_predetection", skymap=skymap, tract=tract, patch=patch, band=band if (band != "z") else "i",
            collections=collection,
            parameters={"bbox": bbox},
        )
        coadds[band] = coadd.image.array
        coadd_wcs_shifted = coadd.wcs.copyAtShiftedPixelOrigin(-Extent2D(coadd.getXY0()))
        psf = coadd.psf.computeKernelImage(coadd.wcs.skyToPixel(coord))
        psfs_hst[band] = gs.InterpolatedImage(gs.Image(psf.array, scale=0.2)).drawImage(scale=0.04).array

    psf_hst_ri = psfs_hst["i"] + psfs_hst["r"]
    psf_hst_ri /= np.sum(psf_hst_ri)
    psfs_hst["ri"] = psf_hst_ri

    coadds["ri"] = coadds["i"] + coadds["r"]

    if not do_cache:
        cutouts_hst_convolved = {}
    for band, cutout_hst in cutouts_hst.items():
        coadd_img = coadds[band]
        fig, ax = plt.subplots(nrows=3)
        if not do_cache or (band not in cutouts_hst_convolved):
            cutout_hst_convolved = fftconvolve(cutout_hst.data, psfs_hst[band], mode="same")
            cutout_hst_convolved[masks_hst[band]] = 0

            cutout_hst_reinterp = reproject_interp(
                (cutout_hst_convolved, wcs_hst),
                fits.Header(coadd_wcs_shifted.getFitsMetadata().toDict()),
                return_footprint=False, shape_out=coadd_img.shape,
            )
            cutouts_hst_convolved[band] = cutout_hst_reinterp
        ax[0].imshow(np.arcsinh(coadd_img), cmap="gray")
        ax[1].imshow(np.arcsinh(cutouts_hst_convolved[band]), cmap="gray")
        ratio = cutouts_hst_convolved[band]/coadd_img
        if do_cache:
            ratios[band] = ratio
        ax[2].imshow(np.clip(ratio, 0, 100), cmap="gray")
        fig.tight_layout()

    img_rgb = make_lupton_rgb(
        *(fac*cutouts_hst[band][::5,::5] for band, fac in (("z", 0.4), ("ri", 0.5), ("g", 1.5))),
        stretch=50, Q=8,
    )
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(np.log10(coadds["ri"]), cmap="gray")
    ax[0].set_title("LSST ri")
    ax[1].imshow(np.log10(cutouts_hst_convolved["z"]), cmap="gray")
    ax[1].set_title("HST F850LP PSF-matched, rewarped")
    fig.tight_layout()
    plt.show()

plt.imshow(img_rgb)
plt.title("HST F850LP,F814W,F475W")
plt.tight_layout()
