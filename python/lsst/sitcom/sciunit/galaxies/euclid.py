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

import glob
from operator import itemgetter
import os
from typing import Any, Iterable

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
import astropy.table
import astropy.units as u
from astropy.wcs import WCS
import logging
from lsst.geom import degrees, SpherePoint
import lsst.skymap
from lsst.sphgeom import ConvexPolygon
import numpy as np

"""
Original query for mosaic table:

SELECT (
    mp.file_name, mp.mosaic_product_oid, mp.tile_index, mp.instrument_name, mp.filter_name, mp.category,
    mp.second_type, mp.ra, mp.dec, mp.technique, mp.stc_s
)
FROM sedm.mosaic_product AS mp
WHERE (release_name='Q1_R1') 
AND ((instrument_name='NISP') OR (instrument_name='VIS')) 
AND (category='SCIENCE') 
AND ((mp.fov IS NOT NULL AND INTERSECTS(CIRCLE('ICRS',53.13,-28.1,1), mp.fov)=1)) 
ORDER BY mp.tile_index ASC
"""

has_astroquery_euclid = False
try:
    from astroquery.esa.euclid import Euclid
    has_astroquery_euclid = True
except ImportError:
    Euclid = None

_log = logging.getLogger("lsst.sitcom.sciunit.galaxies.euclid")

datasettype_default = "euclid_q1_coadd"
filename_format_default = "{datasettype}_{tract}_{patch}_{band}_{skymap}"
path_euclid = (
    f'{os.getenv("SCIUNIT_GALAXIES_EUCLID_DIR", "/sdf/data/rubin/shared/euclid/q1")}'
    f'/{{skymap}}'
)


def get_cutouts_euclid(
    skymap: str,
    tract: int,
    patch: int,
    bands: Iterable[str],
    position: SkyCoord,
    cutout_size: int,
    fits_euclid: dict | None = None,
    keep_fits: bool = False,
) -> tuple[dict[str, Cutout2D], tuple[float, float, float, float]]:
    """Get the Euclid cutouts for a patch in a set of bands.

    Parameters
    ----------
    skymap
        The skymap name.
    tract
        The tract id.
    patch
        The patch id.
    bands
        An iterable of bands.
    position
        The center of the cutout.
    cutout_size
        The size of the cutout to pass to Cutout2D.
    fits_euclid
        A dictionary of references to opened FITS files (HDUList).
    keep_fits
        Whether to store any newly-opened files in fits_euclid.

    Returns
    -------
    cutouts
        The cutout for each band.
    extent
        The RA/dec extent, as expected by matplotlib.pyplot.imshow.
    """
    path_base = f"{path_euclid.format(skymap=skymap)}/{tract}/{patch}"
    if fits_euclid is None:
        fits_euclid = {}
        if keep_fits:
            raise ValueError("Must pass in fits_euclid if keep_fits")

    cutout = None
    cutouts = {}
    for band in bands:
        if (fits_band := fits_euclid.get(band)) is None:
            filename = filename_format_default.format(
                datasettype=datasettype_default, tract=tract, patch=patch, band=band, skymap=skymap,
            )
            filepath = f"{path_base}/{filename}.fits"
            if not os.path.isfile(filepath):
                filepath = f"{path_base}/{filename}.fits.gz"
                if not os.path.isfile(filepath):
                    raise FileNotFoundError(f"{path_base}/{filename}.fits[.gz] is not a file")
            fits_band = fits.open(filepath)
            if keep_fits:
                fits_euclid[band] = fits_band
        n_finite = 0
        cutout_best = None
        image_indices = tuple(hdu.ver for hdu in fits_band if hdu.name == "image")

        for index in image_indices:
            hdu_cdfs = fits_band["image", index]
            wcs_cdfs = WCS(hdu_cdfs)
            try:
                cutout = Cutout2D(
                    hdu_cdfs.data,
                    position=position,
                    size=cutout_size,
                    wcs=wcs_cdfs,
                    copy=True,
                )
            except NoOverlapError:
                continue
            n_finite_new = np.sum(np.isfinite(cutout.data))
            if n_finite_new > n_finite:
                n_finite = n_finite_new
                cutout_best = cutout
        cutout = cutout_best
        cutouts[band] = cutout
    if cutout is not None:
        radec_euclid_begin = cutout.wcs.pixel_to_world(0, 0)
        radec_euclid_end = cutout.wcs.pixel_to_world(cutout.shape[1], cutout.shape[0])
        extent = (
            radec_euclid_begin.ra.value, radec_euclid_end.ra.value, radec_euclid_begin.dec.value,
            radec_euclid_end.dec.value,
        )
    else:
        extent = None
    return cutouts, extent


def make_patch_cutouts(
    mosaic_table: astropy.table.Table,
    filepath: str,
    band: str,
    skymap: lsst.skymap.BaseSkyMap,
    skymap_name: str,
    tracts: Iterable[int],
    save_directory: str,
    datasettype: str | None = None,
    filename_format: str | None = None,
):
    if datasettype is None:
        datasettype = datasettype_default
    if filename_format is None:
        filename_format = filename_format_default

    errors = []
    for dimension in ("datasettype", "skymap", "tract", "patch"):
        if dimension not in filename_format:
            errors.append(f"{dimension} not in filename_format")

    kwargs_format = dict(
        datasettype=datasettype, skymap=skymap_name,
    )
    if "{band}" in filename_format:
        if not band:
            errors.append(f"must specify band")
    if errors:
        raise ValueError(f"Found errors with {filename_format}: {'\n'.join(errors)}")

    band_short, band_type = (band[4:], "NIR") if band.startswith("NIR") else (band, "VIS")
    kwargs_format["band"] = band_short

    mosaic_table = mosaic_table[mosaic_table["filter_name"] == band.upper()]

    polygons_euclid = [
        ConvexPolygon([
            SpherePoint(float(radec[2 * idx]), float(radec[2 * idx + 1]), degrees).getVector()
            for idx in range(4)
        ])
        for radec in (
            stc_s[14:-1].split(" ") for stc_s in mosaic_table["stc_s"]
        )
    ]

    prefix_mosaic = "EUC_MER_MOSAIC-"
    prefix_bgsub = "EUC_MER_BGSUB-MOSAIC-"

    for tract in tracts:
        wcs_tract = skymap[tract].wcs
        tract_path = f"{save_directory}/{tract}"
        if not os.path.exists(tract_path):
            os.mkdir(tract_path)
        kwargs_format["tract"] = str(tract)

        for patchInfo in skymap[tract]:
            patch = patchInfo.getSequentialIndex()
            kwargs_format["patch"] = str(patch)
            poly_inner = patchInfo.getOuterSkyPolygon()
            # Why is there an extra trivial dimension? Who knows.
            matches = np.argwhere([poly_inner.overlaps(poly_euclid) for poly_euclid in polygons_euclid])[:, 0]

            if len(matches) > 0:
                mosaics = mosaic_table["file_name"][matches]
                bbox = patchInfo.getOuterBBox()
                # The factor of 2 is for the 0.1" Euclid pixel scale
                height, width = (x * 2 for x in (bbox.height, bbox.width))
                center = wcs_tract.pixelToSky(*bbox.getCenter())
                center = SkyCoord(center.getRa().asDegrees(), center.getDec().asDegrees(), unit=u.degree)
                hdus = []
                hdus_var = []
                hdus_mask = []

                for idx_mosaic, mosaic_filename in enumerate(mosaics):
                    hdu = fits.open(f"{filepath}/{mosaic_filename}")[0]
                    wcs = WCS(hdu)

                    try:
                        cutout = Cutout2D(
                            data=hdu.data, position=center, size=(height, width),
                            wcs=wcs, copy=False, mode="trim",
                        )
                        header = hdu.header.copy()
                        header.update(cutout.wcs.to_header())
                        cutout.data[np.isfinite(cutout.data)] /= (header["MAGZERO"] * u.ABmag).to(u.nJy).value

                        del header["MAGZERO"]
                        header["BUNIT"] = "nJy"
                        header["EXTNAME"] = "image"
                        header["EXTVER"] = idx_mosaic

                        hdus.append((fits.ImageHDU(data=cutout.data, header=header), cutout.data.size))

                        band_tile = mosaic_filename.split(prefix_bgsub, 1)[1]
                        if band_tile[:3] == "NIR":
                            band_prefix = band_tile[:4]
                            band_tile = band_tile[4:]
                        else:
                            band_prefix = ""
                        band_short_file, tile = band_tile.split("-", 1)[0].split("_")
                        assert band_short_file == band_short
                        band_file = f"{band_prefix}{band_short}"

                        filename_rms = glob.glob(f"{filepath}/{prefix_mosaic}{band_file}-RMS_{tile}*.fits")[0]
                        hdu = fits.open(filename_rms)[0]

                        cutout = Cutout2D(
                            data=hdu.data, position=center, size=(height, width),
                            wcs=wcs, copy=False, mode="trim",
                        )
                        header = hdu.header.copy()
                        header.update(cutout.wcs.to_header())
                        isfinite = np.isfinite(cutout.data)
                        cutout.data[isfinite] /= (header["MAGZERO"] * u.ABmag).to(u.nJy).value
                        cutout.data[isfinite] *= cutout.data[isfinite]

                        del header["MAGZERO"]
                        header["BUNIT"] = "nJy**2"
                        header["EXTNAME"] = "variance"
                        header["EXTVER"] = idx_mosaic
                        hdus_var.append((fits.ImageHDU(data=cutout.data, header=header), cutout.data.size))

                        filename_flag = glob.glob(f"{filepath}/{prefix_mosaic}{band_file}-FLAG_{tile}*.fits.gz")[0]
                        hdu = fits.open(filename_flag)[0]
                        cutout = Cutout2D(
                            data=hdu.data, position=center, size=(height, width),
                            wcs=wcs, copy=False, mode="trim",
                        )
                        header = hdu.header.copy()
                        header.update(cutout.wcs.to_header())
                        header["EXTNAME"] = "mask"
                        header["EXTVER"] = idx_mosaic
                        hdus_mask.append((fits.ImageHDU(data=cutout.data, header=header), cutout.data.size))

                    except NoOverlapError as e:
                        raise RuntimeError(f"{tract=} {patch=} {mosaic_filename=} got {e=}")

                if hdus:
                    header = fits.Header()
                    header["LSST BUTLER DATASETTYPE"] = datasettype
                    header["LSST BUTLER DATAID SKYMAP"] = skymap_name
                    header["LSST BUTLER DATAID TRACT"] = tract
                    header["LSST BUTLER DATAID PATCH"] = patch

                    hdu_primary = fits.PrimaryHDU(header=header)
                    for hdulist in (hdus, hdus_var, hdus_mask):
                        hdulist.sort(key=itemgetter(1), reverse=True)

                    cutout_path = f"{tract_path}/{patch}"
                    if not os.path.exists(cutout_path):
                        os.mkdir(cutout_path)
                    cutout_filename = filename_format.format(**kwargs_format)
                    _log.info(f"Writing {len(hdus)} to {cutout_filename}")
                    hdus_all = [hdu_primary] + [
                        hdu[0] for hdulist in (hdus, hdus_var, hdus_mask) for hdu in hdulist
                    ]
                    fits.HDUList(hdus_all).writeto(
                        f"{cutout_path}/{cutout_filename}.fits.gz", overwrite=True,
                    )


def query_tract_catalog(
    tractInfo: lsst.skymap.tractInfo.TractInfo,
    name_skymap: str,
    n_patches: int = 5,
    tmpFile: str | None = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> tuple[astropy.table.Table | None, list[str], list[Any]]:
    """Query Euclid for a tract-level catalogs.

    Parameters
    ----------
    tractInfo
        The tract to make a catalog for.
    name_skymap
        The name of the skymap the tract is from.
    n_patches
        The number of "patches" per axis to split the tract into for querying.
        This does not need to match the skymap's patch size and is needed only
        for submitting request without authentication as there is a file size
        limit.
    tmpFile
        A filename to save downloaded patch catalogs to. May contain
        {idx_patch}, which will add the patch index to the filename.
        If None, queries will attempt to go directly to memory.
    skip_existing
        Look for existing tmp files and try to load them.
    verbose


    Returns
    -------

    """
    columns_m = [
        "object_id",
        "right_ascension",
        "declination",
        "right_ascension_psf_fitting",
        "declination_psf_fitting",
        "vis_det",
    ]

    columns_flux = []

    bands_vis = ("vis",)
    bands_nir = ("y", "j", "h",)
    bands_all = bands_vis + bands_nir

    for algos, bands in (
        (("psf",), bands_vis),
        (("templfit",), bands_nir),
        (("sersic",), bands_all),
    ):
        columns_flux.extend([
            f"flux{suffix}_{band}_{algo}" for suffix in ("", "err") for band in bands for algo in algos
        ])

    columns_band_other = [
        f"flag_{band}" for band in ("vis", "y", "j", "h")
    ]
    columns_other = [
        "deblended_flag",
        "parent_id",
        "parent_visnir",
        "blended_prob",
        "she_flag",
        "variable_flag",
        "binary_flag",
        "point_like_flag",
        "point_like_prob",
        "extended_flag",
        "extended_prob",
        "spurious_flag",
        "spurious_prob",
        "mag_stargal_sep",
        "det_quality_flag",
        "mu_max",
        "mumax_minus_mag",
        "segmentation_area",
        "semimajor_axis",
        "semimajor_axis_err",
        "position_angle",
        "position_angle_err",
        "ellipticity",
        "ellipticity_err",
        "kron_radius",
        "kron_radius_err",
        "fwhm",
        "gal_ebv",
        "gal_ebv_err",
        "gaia_id",
        "gaia_match_quality",
    ]

    columns_m = columns_m + columns_flux + columns_band_other + columns_other
    columns_morph = [
        "agn_no",
        "agn_yes",
        "asymmetry",
        "asymmetry_err",
        "bar_no",
        "bar_strong",
        "bar_weak",
        "basic_download_data_oid",
        "disk_sersic_angle",
        "disk_sersic_angle_err",
        "disk_sersic_disk_axis_ratio",
        "disk_sersic_disk_axis_ratio_err",
        "disk_sersic_disk_radius",
        "disk_sersic_disk_radius_err",
        "disk_sersic_duration",
        "disk_sersic_flags",
        "disk_sersic_iterations",
        "disk_sersic_reduced_chi2",
        "disk_sersic_sersic_axis_ratio",
        "disk_sersic_sersic_axis_ratio_err",
        "disk_sersic_sersic_index",
        "disk_sersic_sersic_index_err",
        "disk_sersic_sersic_radius",
        "disk_sersic_sersic_radius_err",
        "etg_or_ltg",
        "major_merger",
        "peculiar_no",
        "peculiar_yes",
        "ring_no",
        "ring_yes",
        "sersic_angle",
        "sersic_angle_err",
        "sersic_ext_duration",
        "sersic_ext_flags",
        "sersic_ext_iterations",
        "sersic_ext_reduced_chi2",
        "sersic_sersic_nir_axis_ratio",
        "sersic_sersic_nir_axis_ratio_err",
        "sersic_sersic_nir_index",
        "sersic_sersic_nir_index_err",
        "sersic_sersic_nir_radius",
        "sersic_sersic_nir_radius_err",
        "sersic_sersic_vis_axis_ratio",
        "sersic_sersic_vis_axis_ratio_err",
        "sersic_sersic_vis_index",
        "sersic_sersic_vis_index_err",
        "sersic_sersic_vis_radius",
        "sersic_sersic_vis_radius_err",
        "sersic_visnir_duration",
        "sersic_visnir_flags",
        "sersic_visnir_iterations",
        "sersic_visnir_reduced_chi2",
        "t_type",
        "gini",
        "gini_err",
        "moment_20",
        "moment_20_err",
        "concentration",
        "concentration_err",
        "smoothness",
        "smoothness_err",
    ]
    columns_pz = [
        f"flux{suffix}_{band}_{algo}" for suffix in ("", "err") for band in bands_all for algo in ("unif",)
    ]
    columns_pc = ["phz_classification",]

    columns_all = [f"m.{c}" for c in columns_m] + [f"mor.{c}" for c in columns_morph] + [
        f"pz.{c}" for c in columns_pz] + [f"pc.{c}" for c in columns_pc]

    query = (
        f"SELECT {', '.join(columns_all)}"
        " FROM catalogue.mer_catalogue AS m"
        " LEFT OUTER JOIN catalogue.mer_morphology AS mor USING (object_id)"
        " LEFT OUTER JOIN catalogue.phz_photo_z AS pz USING (object_id)"
        " LEFT OUTER JOIN catalogue.phz_classification AS pc USING (object_id)"
        " WHERE ("
        "(m.right_ascension > {ra_min}) AND (m.right_ascension < {ra_max})"
        " AND (m.declination > {dec_min}) AND (m.declination < {dec_max})"
        ")"
    )

    corners = tractInfo._innerBoxCorners
    (ra_min, dec_min), (ra_max, dec_max) = ((c.getLon().asDegrees(), c.getLat().asDegrees()) for c in corners)
    delta = 1e-12

    ra_ranges = np.linspace(ra_min - delta, ra_max + delta, n_patches + 1)
    dec_ranges = np.linspace(dec_min - delta, dec_max + delta, n_patches + 1)

    jobs = []
    queries = []
    tables = []

    has_index = (tmpFile is not None) and ("idx_patch" in tmpFile)
    idx_patch = 0

    for dec_max in dec_ranges[1:]:
        for ra_max in ra_ranges[1:]:
            query_patch = query.format(ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max)
            if has_astroquery_euclid:
                filename = None if tmpFile is None else (
                    tmpFile.format(idx_patch=idx_patch) if has_index else tmpFile)
                if skip_existing and (filename is not None) and os.path.isfile(filename):
                    _log.info(f"Loading {idx_patch=} from {filename=}")
                    objects = astropy.table.Table.read(filename)
                else:
                    _log.info(f"Launching query for {idx_patch=} to {filename=}")
                    job = Euclid.launch_job(
                        query_patch, verbose=verbose, output_format="votable",
                        dump_to_file=tmpFile is not None, output_file=filename,
                    )
                    jobs.append(job)
                    objects = job.get_results() if (job is not None) else None

                if objects is not None:
                    if "object_id_2" in objects.colnames:
                        del objects["object_id_2"]
                    _log.info(f"Creating {name_skymap=} patch column for {idx_patch=}")
                    coords = [
                        SpherePoint(ra, dec, degrees)
                        for ra, dec in zip(objects["right_ascension"], objects["declination"])
                    ]
                    within = np.array([tractInfo.contains(coord) for coord in coords])
                    if np.sum(within) != len(within):
                        objects = objects[within]
                        coords = [coord for coord, in_tract in zip(coords, within) if in_tract]
                    patches = np.array(
                        [tractInfo.findPatch(coord).getSequentialIndex() for coord in coords],
                        dtype=np.int16,
                    )
                    objects["patch"] = patches
                    tables.append(objects)
            ra_min = ra_max
            queries.append(query_patch)
            idx_patch += 1
        dec_min = dec_max
        ra_min = ra_ranges[0]

    if tables:
        table = astropy.table.vstack(tables)
        table["patch"].description = f"{name_skymap} patch index"
    else:
        table = None

    return table, queries, jobs