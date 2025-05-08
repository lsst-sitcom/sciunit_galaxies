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

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

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
