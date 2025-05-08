# This file is part of sciunit_galaxies.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os.path

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.wcs import WCS

import astropy.units as u
import lsst.geom as geom
import lsst.skymap
import numpy as np

__all__ = ("get_cutouts_cdfs_hst", "path_cdfs_hst", "scale_cdf_hst_asec")

path_cdfs_hst = "/sdf/data/rubin/shared/hst/ecdfs/{skymap}"
scale_cdf_hst_asec = 0.03


def get_cutouts_cdfs_hst(
    tract, patch, bands, skymap, position: SkyCoord, cutout_size, fits_cdfs=None, keep_fits=False,
):
    path_base = f"{path_cdfs_hst.format(skymap=skymap)}/{tract}/{patch}"
    if fits_cdfs is None:
        fits_cdfs = {}
        if keep_fits:
            raise ValueError("Must pass in fits_cdfs if keep_fits")

    cutout = None
    cutouts = {}
    for band in bands:
        if (fits_band := fits_cdfs.get(band)) is None:
            fits_band = fits.open(f"{path_base}/hlf_hst_coadd_{tract}_{patch}_{band}_{skymap}.fits.gz")
            if keep_fits:
                fits_cdfs[band] = fits_band
        hdu_cdfs = fits_band[1]
        wcs_cdfs = WCS(hdu_cdfs)
        cutout = Cutout2D(
            hdu_cdfs.data,
            position=position,
            size=cutout_size,
            wcs=wcs_cdfs,
            copy=True,
        )
        cutouts[band] = cutout
    if cutout is not None:
        radec_hst_begin = cutout.wcs.pixel_to_world(0, 0)
        radec_hst_end = cutout.wcs.pixel_to_world(cutout.shape[1], cutout.shape[0])
        extent_hst = (
            radec_hst_begin.ra.value, radec_hst_end.ra.value, radec_hst_begin.dec.value,
            radec_hst_end.dec.value,
        )
    else:
        extent_hst = None
    return cutouts, extent_hst


def get_hdu_corners(
    data_shape: np.ndarray,
    wcs: WCS,
) -> tuple[geom.SpherePoint, geom.SpherePoint, geom.SpherePoint, geom.SpherePoint]:
    """Get the corners of an image given its dimensions and WCS.

    Parameters
    ----------
    data_shape
        The shape of the data array.
    wcs
        The WCS of the image.

    Returns
    -------
    corners
        The bottom left, bottom right, top right and top left corners.
    """
    x_dim, y_dim = data_shape

    # last argument is for zero-based indices instead of the default 1 for FITS
    bottom_left = wcs.all_pix2world(0, 0, 0)
    bottom_right = wcs.all_pix2world(x_dim -1, 0, 0)
    top_right = wcs.all_pix2world(x_dim - 1, y_dim - 1, 0)
    top_left = wcs.all_pix2world(0, y_dim - 1, 0)

    return (
        geom.SpherePoint(bottom_left[0], bottom_left[1], geom.degrees),
        geom.SpherePoint(bottom_right[0], bottom_right[1], geom.degrees),
        geom.SpherePoint(top_right[0], top_right[1], geom.degrees),
        geom.SpherePoint(top_left[0], top_left[1], geom.degrees),
    )


def make_patch_cutouts(
    image_filepath: str,
    skymap: lsst.skymap.BaseSkyMap,
    skymap_name: str,
    save_directory: str,
    datasettype: str | None = None,
    filename_format: str | None = None,
    band: str | None = None,
    weight_filepath: str | None = None,
    is_weight_variance: bool = False,
):
    """Make patch-based cutouts from a single HLF HST image.

    Parameters
    ----------
    image_filepath
        The filename of the image.
    skymap
        The skymap object.
    skymap_name
        The name of the skymap.
    save_directory
        The path to write the output files to.
    datasettype
        The butler dataset type name.
    filename_format
        The format of the filename, which must include {datasettype},
        {skymap}, {tract}, {patch}, and may include {band}.
    band
        The band (filter) of the image.
    weight_filepath
        The path to the weight image.
    is_weight_variance
        Whether the weight image is variance (True) or inverse variance
        (False).
    """
    if datasettype is None:
        datasettype = "hlf_hst_coadd"
    if filename_format is None:
        filename_format = "{datasettype}_{tract}_{patch}_{band}_{skymap}.fits.gz"

    errors = []
    for dimension in ("datasettype", "skymap", "tract", "patch"):
        if dimension not in filename_format:
            errors.append(f"{dimension} not in filename_format")

    kwargs_format = dict(
        datasettype=datasettype, skymap=skymap_name,
    )
    if "{band}" in filename_format:
        if band is None:
            errors.append(f"must specify band")
        else:
            kwargs_format["band"] = band
    if errors:
        raise ValueError(f"Found errors with {filename_format}: {'\n'.join(errors)}")

    # read in image
    with fits.open(image_filepath) as hdu:
        image_header = hdu[0].header
        image_header["EXTNAME"] = "image"
        data = hdu[0].data
        wcs = WCS(image_header)
        if (zeropoint := image_header.get("ZEROPNT")) is not None:
            bunit = (zeropoint * u.ABmag).to(u.nJy).value
        else:
            bunit = None
        if bunit:
            image_header["BUNIT"] = "nJy"

    headers = [image_header]
    if weight_filepath is not None:
        with fits.open(weight_filepath) as hdu:
            variance_header = hdu[0].header
            variance_header["EXTNAME"] = "variance"
            variance = hdu[0].data
            if not is_weight_variance:
                variance = 1/variance
            if bunit is not None:
                variance_header["BUNIT"] = "nJy**2"
        headers.append(variance_header)
    else:
        variance, variance_header = None, None

    for header in headers:
        for key in ("EXPTIME", "ZEROPNT"):
            if key in header:
                del header[key]

    # read in cutout info
    corners = get_hdu_corners(data.shape, wcs)

    tractInfos = [skymap.findTract(corner) for corner in corners]

    # TODO: handle case of >4 tract overlap (need more than corners)
    tractInfos = {tractInfo.tract_id: tractInfo for tractInfo in tractInfos}

    for tract, tractInfo in tractInfos.items():
        kwargs_format["tract"] = str(tract)
        wcs_tract = tractInfo.wcs
        tract_path = f"{save_directory}/{tract}"
        if not os.path.exists(tract_path):
            os.mkdir(tract_path)

        for patchInfo in tractInfo:
            patch = patchInfo.getSequentialIndex()
            kwargs_format["patch"] = str(patch)
            bbox = patchInfo.getOuterBBox()
            # HST mosaic pixel scale (0.03) to Rubin (0.2) = 20/3
            # This would need to change if we wanted to use 0.06" mosaics
            # ... but I don't see any reason to do so
            height, width = (int(np.ceil((x*20)/3)) for x in (bbox.height, bbox.width))
            center = wcs_tract.pixelToSky(*bbox.getCenter())
            center = SkyCoord(center.getRa().asDegrees(), center.getDec().asDegrees(), unit=u.degree)
            try:
                image_cutout = Cutout2D(
                    data=data, position=center, size=(height, width),
                    wcs=wcs, copy=True, mode="partial", fill_value=np.nan,
                )
                if bunit is not None:
                    image_cutout.data *= bunit
            except NoOverlapError:
                continue

            if not (np.sum(np.isfinite(image_cutout.data)) > 0):
                continue

            if variance is not None:
                variance_cutout = Cutout2D(
                    data=variance, position=center, size=(height, width),
                    wcs=wcs, copy=True, mode="partial", fill_value=np.nan,
                )
                if bunit is not None:
                    variance_cutout.data *= bunit ** 2

            # create new HDU and update header values
            header = fits.Header()
            header["LSST BUTLER DATASETTYPE"] = datasettype
            header["LSST BUTLER DATAID SKYMAP"] = skymap_name
            header["LSST BUTLER DATAID TRACT"] = tract
            header["LSST BUTLER DATAID PATCH"] = patch
            cutout_hdus = [
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(data=image_cutout.data, header=image_header),
            ]
            if variance is not None:
                variance_hdu = fits.ImageHDU(data=variance_cutout.data, header=variance_header)
                cutout_hdus.append(variance_hdu)

            for hdu in cutout_hdus[1:]:
                hdu.header.update(image_cutout.wcs.to_header())

            cutout_path = f"{tract_path}/{patch}"
            if not os.path.exists(cutout_path):
                os.mkdir(cutout_path)
            cutout_filename = filename_format.format(**kwargs_format)
            fits.HDUList(cutout_hdus).writeto(f"{cutout_path}/{cutout_filename}", overwrite=True)
