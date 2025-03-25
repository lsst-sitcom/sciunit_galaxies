import os.path

from astropy.nddata import Cutout2D, NoOverlapError
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import lsst.geom as geom
import lsst.skymap
import numpy as np


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
    """Make patch-based cutouts from a single HST iamge.

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
