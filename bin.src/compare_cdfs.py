import math

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u
import galsim as gs
import lsst.afw.image
import lsst.daf.butler as dafButler
from lsst.geom import Point2D
import numpy as np
from skimage.transform import resize


def calibrate_exposure(exposure: lsst.afw.image.Exposure) -> lsst.afw.image.MaskedImageF:
    calib = exposure.getPhotoCalib()
    image = calib.calibrateImage(
        lsst.afw.image.MaskedImageF(exposure.image, mask=exposure.mask, variance=exposure.variance)
    )
    return image


tract = 5063

name_skymap = "lsst_cells_v1"
butler = dafButler.Butler(
    "/repo/embargo",
    collections=["LSSTComCam/runs/nightlyValidation/20241108/d_2024_11_05/DM-47059"],
)
skymap = butler.get("skyMap", skymap=name_skymap, collections="skymaps")
tractInfo = skymap[tract]
patches = np.arange(np.prod(tractInfo.getNumPatches()))
wcs = tractInfo.wcs

bands_hst_lsst = {
    "F435W": ("g",),
    "F606W": ("r",),
    "F775W": ("i",),
    "F814W": ("i", "z"),
}

scale_lsst = 0.2
scale_hst = 0.03

shift_x, shift_y = -1.088*scale_hst/scale_lsst, 1.165*scale_hst/scale_lsst

path_cdfs_hst = "/sdf/data/rubin/user/dtaranu/tickets/cdfs/"

coadds_lsst_patch = {}
coadds_hst_patch = {}

for band_hst, bands_lsst in bands_hst_lsst.items():
    img_cdfs = fits.open(
        f"{path_cdfs_hst}hlsp_hlf_hst_acs-30mas_goodss_{band_hst.lower()}_v2.0_sci.fits.gz"
    )
    wcs_cdfs = WCS(img_cdfs[0])

    for patch in patches:
        patchInfo = tractInfo[patch]
        bbox_outer = patchInfo.outer_bbox

        (ra_begin, dec_begin), (ra_end, dec_end) = (
            (x.getRa().asDegrees(), x.getDec().asDegrees())
            for x in (
                wcs.pixelToSky(*(bbox_outer.getBegin())),
                wcs.pixelToSky(*(bbox_outer.getEnd())),
            )
        )

        center = SkyCoord((ra_end + ra_begin) / 2., (dec_end + dec_begin) / 2., unit=u.degree)
        height_pix, width_pix = bbox_outer.getHeight(), bbox_outer.getWidth()

        coadd_hst_full = Cutout2D(
            img_cdfs[0].data,
            position=center,
            size=(
                int(math.ceil(width_pix*scale_lsst/scale_hst)),
                int(math.ceil(height_pix*scale_lsst/scale_hst)),
            ),
            wcs=wcs_cdfs,
            copy=True,
        )
        coadd_hst_full.data *= 10**(((u.nJy).to(u.ABmag) - img_cdfs[0].header["ZEROPNT"])/2.5)

        coadds_lsst = {
            band: butler.get("deepCoadd_calexp", tract=tract, patch=patch, band=band, skymap=name_skymap)
            for band in bands_lsst
        }
        psfs_lsst = {}
        for band, coadd in coadds_lsst.items():
            psfs_lsst[band] = coadd.psf
            coadds_lsst[band] = calibrate_exposure(coadd)

        n_bands_lsst = len(coadds_lsst)
        coadd_lsst = coadds_lsst[bands_lsst[0]].image.array
        for band in bands_lsst[1:]:
            coadd_lsst += coadds_lsst[band].image.array
        coadd_hst_downsampled = resize(coadd_hst_full.data, output_shape=coadd_lsst.shape)*(
            scale_lsst/scale_hst)**2
        mask_inv = coadd_hst_downsampled != 0

        coadd_hst = np.zeros_like(coadd_lsst)

        y_max, x_max = coadd_lsst.shape
        y_max_hst, x_max_hst = coadd_hst_full.data.shape

        cell_size_hst, cell_size_lsst = 1000, 150
        pixels_crop = 15
        cell_iter = cell_size_lsst - 2*pixels_crop
        pixels_crop_hst = round(pixels_crop * scale_lsst / scale_hst)
        cell_iter_hst = cell_size_hst - 2*pixels_crop_hst

        x_begin_patch, y_begin_patch = bbox_outer.getBeginX(), bbox_outer.getBeginY()

        x_begin, y_begin = 0, 0
        x_begin_hst, y_begin_hst = 0, 0

        while x_begin < x_max:
            x_end = min(x_begin + cell_size_lsst, x_max)
            x_end_hst = min(x_begin_hst + cell_size_hst, x_max_hst)

            while y_begin < y_max:
                y_end = min(y_begin + cell_size_lsst, y_max)
                y_end_hst = min(y_begin_hst + cell_size_hst, y_max_hst)

                point_patch = Point2D(x_begin + x_begin_patch, y_begin + y_begin_patch)

                psf_lsst = psfs_lsst[bands_lsst[0]].computeKernelImage(point_patch).array
                for band in bands_lsst[1:]:
                    psf_lsst += psfs_lsst[band].computeKernelImage(point_patch).array
                psf_lsst /= n_bands_lsst

                img_psf = gs.InterpolatedImage(gs.Image(psf_lsst, scale=scale_lsst))
                cutout_hst = coadd_hst_full.data[y_begin_hst:y_end_hst, x_begin_hst:x_end_hst]
                if (cutout_hst > 0).any():
                    cutout_hst_gs = gs.InterpolatedImage(gs.Image(cutout_hst, scale=scale_hst))
                    img_convolved = gs.Convolve(cutout_hst_gs, img_psf).shift(shift_x, shift_y).drawImage(
                        nx=cell_size_lsst, ny=cell_size_lsst).array
                    coadd_hst[
                        y_begin + pixels_crop:y_end - pixels_crop, x_begin + pixels_crop:x_end - pixels_crop
                    ] = img_convolved[
                        pixels_crop:y_end - y_begin - pixels_crop,
                        pixels_crop:x_end - x_begin - pixels_crop
                    ]
                else:
                    print(f"{x_begin=}, {y_begin=} has no data")

                y_begin += cell_iter
                y_begin_hst += cell_iter_hst

            x_begin += cell_iter
            x_begin_hst += cell_iter_hst
            y_begin, y_begin_hst = 0, 0
            print(x_begin)



"""
    if plot:
        coadd_lsst = calibrate_exposure(
            butler.get("deepCoadd_calexp", tract=tract, patch=patch, band="i", skymap=name_skymap),
        )
        x_start_lsst = patchInfo.outer_bbox.getHeight() - patchInfo.inner_bbox.getHeight()

        psf = coadd_lsst.psf.computeKernelImage(patchInfo.outer_bbox.getCenter())
        img_psf = gs.InterpolatedImage(gs.Image(psf.array, scale=scale_lsst))
        cutout_hst_gs = gs.InterpolatedImage(gs.Image(cutout.data, scale=scale_hst))
        img_convolved = gs.Convolve(cutout_hst_gs, img_psf).drawImage(
            nx=cutout.data.shape[1], ny=cutout.data.shape[0])

bands = ("i", "r", "g")
if plot:

import galsim as gs
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
import matplotlib as mpl

mpl.rcParams.update({"image.origin": "lower", 'font.size': 13})

    weight_mean = np.mean([bands_weights_lsst[band] for band in bands])
    for band in bands:
        bands_weights_lsst[band] /= weight_mean
    abs_mag_hst = {
        "F435W": 5.35,
        "F606W": 4.72,
        "F775W": 4.52,
        "F814W": 4.52,
    }
    weights_hst = 1/(u.ABmag.to(u.Jy, [abs_mag_hst[band] for band in bands_hst]))
    weights_hst_mean = np.mean(weights_hst)
    bands_weights_hst = {
        band: weight_hst/weights_hst_mean for band, weight_hst in zip(bands_hst, weights_hst)
    }
    kwargs_lup = dict(minimum=-1, Q=8, stretch=100)
    kwargs_lup_hst = dict(minimum=-0.3, Q=8, stretch=4)

"""
