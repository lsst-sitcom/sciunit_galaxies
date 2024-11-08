from functools import cached_property
import glob

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from lsst.afw.geom import SkyWcs
from lsst.afw.image import ExposureF
from lsst.afw.table import SourceCatalog
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, pq
import lsst.gauss2d.fit as g2f
from lsst.meas.extensions.multiprofit.fit_coadd_multiband import CatalogExposurePsfs
from lsst.meas.extensions.multiprofit.plots import ObjectTableBase
from lsst.multiprofit import ComponentGroupConfig, SersicComponentConfig, ModelConfig, SourceConfig
from lsst.multiprofit.fitting import CatalogSourceFitterConfigData
from lsst.sitcom.sciunit.galaxies.fit_cosmos_hst import CatalogExposureCosmosHstBase
from lsst.sitcom.sciunit.galaxies.fit_cosmos_hst_objects import (
    CatalogExposureCosmosHstObjects, MultiProFitCosmosHstObjectsConfig, MultiProFitCosmosHstObjectsFitter,
)
from lsst.sitcom.sciunit.galaxies.fit_cosmos_hst_stars import CatalogExposureCosmosHstStars
import numpy as np
import pydantic


def get_dataset_filepath(dataset: str, tract: int = 9813, patch: int = 40, band=None, suffix: str = ".parq"):
    patch_dir = f"{patch}/" if patch else ''
    filedir = f"{dataset}/{tract}/{patch_dir}"
    suffix_band = ""
    if band is not None:
        filedir = f"{filedir}{band}/"
        suffix_band = f"_{band}"
    patch_str = f"_{patch}" if patch else ""
    filename = f"{filedir}{dataset}_{tract}{patch_str}{suffix_band}{suffix}"
    return filename


def get_dataset_filename_first(prefix_path: str, **kwargs):
    filepath = get_dataset_filepath(**kwargs)
    filename = next(iter(glob.glob(f"{prefix_path}/{filepath}")))
    return filename


def get_psf_model_fits_filepath(tract: int = 9813, patch: int = 40, band="F814W", kwargs_first: dict = None):
    if kwargs_first is None:
        kwargs_first = {}
    dataset = f"deepCoadd_psfs_multiprofit"
    return (get_dataset_filename_first if kwargs_first else get_dataset_filepath)(
        dataset=dataset, tract=tract, patch=patch, band=band, **kwargs_first
    )


def get_deblended_model_fits_filepath(model: str = "sersic", tract: int = 9813, patch: int = 40):
    dataset = f"cosmos_hst_deblended_{model}_multiprofit"
    return get_dataset_filepath(dataset=dataset, tract=tract, patch=patch)


def get_tiles_from_patch(
    tile_table: Table,
    radec_min: SkyCoord,
    radec_max: SkyCoord,
    path_tiles: str = "",
):
    """Get a list of HST tile filenames covering a patch from a path like
       TESTDATA_COSMOS.

    Parameters
    ----------
    tile_table
        The tile table containing path_[science/weight] and [ra/dec]_[min/max]
        columns.
    radec_min
        The sky coordinate of the minimum of the area to cover.
    radec_max
        The sky coordinate of the maximum of the area to cover, where both
        RA and dec should be greater than radec_min.
    path_tiles
        A prefix to add to any path_[science/weight] fields.

    Returns
    -------
    tiles_within
        A list of tuples of the science and weight image file path for the
    """
    tiles_within = [
        (f'{path_tiles}{row["path_science"]}', f'{path_tiles}{row["path_weight"]}') for row in tile_table[
            ((tile_table["ra_min"] < radec_max.ra.value) & (tile_table["ra_max"] > radec_min.ra.value))
            & ((tile_table["dec_min"] < radec_max.dec.value) & (tile_table["dec_max"] > radec_min.dec.value))
            ]
    ]
    return tiles_within


def get_tiles_from_patch_path(
    testdata_path: str,
    radec_min: SkyCoord,
    radec_max: SkyCoord,
):
    """As get_tiles_from patch, but taking a path to TESTDATA_COSMOS."""
    tile_table = Table.read(f"{testdata_path}/cosmos_hst_tiles.ecsv")
    path_tiles = f"{testdata_path}/cosmos_hst_tiles/"
    return get_tiles_from_patch(
        tile_table=tile_table, radec_min=radec_min, radec_max=radec_max, path_tiles=path_tiles,
    )


def read_data_hsc(
    testdata_path: str,
    bands: list[str] | None = None,
    tract: int = 9813,
    patch: int = 40,
) -> tuple[dict[str, tuple[SourceCatalog, ExposureF]], SourceCatalog]:
    """Read COSMOS HSC data from a path like TESTDATA_COSMOS and return
       a gauss2d_fit Observation of it.

    Parameters
    ----------
    testdata_path
        The path to load from.
    bands
        A list of bands to load catalog/exposure pairs for.'
    tract
        The tract to load.
    patch
        The patch to load.

    Returns
    -------
    catexps
        A dict of catalog-exposure pairs by band.
    ref
        The ref catalog containing column values in the reference band for
        each object.
    """

    if bands is None:
        bands = tuple()

    if len(set(bands)) != len(bands):
        raise ValueError(f"{bands=} must be a set")
    kwargs_path = {"tract": tract, "patch": patch, "suffix": "*.fits", "prefix_path": f"{testdata_path}/"}
    catexps = {}
    for band in bands:
        exposure = ExposureF.readFits(
            get_dataset_filename_first(dataset="deepCoadd_calexp", band=band, **kwargs_path)
        )
        catalog = SourceCatalog.readFits(
            get_dataset_filename_first(dataset="deepCoadd_meas", band=band, **kwargs_path)
        )
        catexps[band] = (catalog, exposure)
    ref = SourceCatalog.readFits(get_dataset_filename_first(dataset="deepCoadd_ref", **kwargs_path))
    return catexps, ref


def build_fit_inputs(
    testdata_cosmos_dir: str,
    bands_hsc: tuple[str] | list[str],
    tract: int, patch: int,
    band_hsc_ref: str = "r", band_hst: str = "F814W"
):
    channels_hsc = {band: g2f.Channel.get(band) for band in bands_hsc}

    catexps_hsc, catalog_ref_hsc = read_data_hsc(testdata_path=testdata_cosmos_dir, bands=bands_hsc)
    wcs_hsc = catexps_hsc[band_hsc_ref][1].wcs

    fitter = MultiProFitCosmosHstObjectsFitter(wcs=wcs_hsc)
    config_fit = MultiProFitCosmosHstObjectsConfig(
        config_model=ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            components_sersic={
                                "sersic": SersicComponentConfig(),
                            }
                        )
                    }
                )
            }
        ),
        flag_errors={v: str(k) for k, v in fitter.errors_expected.items()},
        fit_isolated_only=True,
        apply_centroid_pixel_offset=False,
    )

    for band, catexp in catexps_hsc.items():
        table_psf_fits_hsc = arrow_to_astropy(pq.read_table(
            get_psf_model_fits_filepath(
                tract=tract, patch=patch, band=band,
                kwargs_first={"suffix": "*.parq", "prefix_path": f"{testdata_cosmos_dir}/"},
            )
        ))
        catexp = CatalogExposurePsfs(
            catalog=catexp[0],
            exposure=catexp[1],
            channel=channels_hsc[band],
            config_fit=config_fit,
            dataId={"tract": tract, "patch": patch, "band": band},
            table_psf_fits=table_psf_fits_hsc,
            use_sky_coords=True,
        )
        catexps_hsc[band] = catexp

    ra_hsc, dec_hsc = (catalog_ref_hsc[col] for col in ("coord_ra", "coord_dec"))
    radec_min, radec_max = tuple(
        SkyCoord(*((f(coord) * u.rad).to(u.deg) for coord in (ra_hsc, dec_hsc)))
        for f in (np.min, np.max)
    )

    tiles_within = get_tiles_from_patch_path(
        testdata_path=testdata_cosmos_dir, radec_min=radec_min, radec_max=radec_max,
    )

    catalog_hst = arrow_to_astropy(pq.read_table(f"{testdata_cosmos_dir}/cosmos_acs_iphot_200709.parq"))
    catalog_hst["index_row"] = np.arange(len(catalog_hst))
    within_catalog = (catalog_hst["ra"] > radec_min.ra) & (catalog_hst["ra"] < radec_max.ra) & (
        catalog_hst["dec"] > radec_min.dec) & (catalog_hst["dec"] < radec_max.dec)
    catalog_hst = catalog_hst[within_catalog]
    table_psf_fits_hst = arrow_to_astropy(pq.read_table(
        f"{testdata_cosmos_dir}/{get_psf_model_fits_filepath(tract=tract, patch=patch, band=band_hst)}"
    ))
    CatalogExposureCosmosHstStars.adjust_psf_table_values(table_psf_fits_hst, band=band_hst)

    obs_hst, wcs_hst, cos_dec_hst = CatalogExposureCosmosHstBase.get_observation_hst(
        tiles_within, radec_min=radec_min, radec_max=radec_max, band=band_hst,
    )

    catexp_hst = CatalogExposureCosmosHstObjects(
        catalog_hst=catalog_hst,
        catalog_ref_hsc=catalog_ref_hsc,
        config_fit=config_fit,
        cos_dec_hst=cos_dec_hst,
        dataId={"tract": 9813, "patch": patch, "band": band_hst},
        observation_hst=obs_hst,
        table_psf_fits=table_psf_fits_hst,
        wcs_ref=wcs_hsc,
        wcs_hst=wcs_hst,
    )
    channels_all = list(channels_hsc.values()) + [catexp_hst.channel]
    config_fit.bands_fit = [channel.name for channel in channels_all]
    # TODO: Use CatalogSourceFitterCosmosConfigData when ready
    config_data = CatalogSourceFitterConfigData(
        config=config_fit,
        channels=channels_all,
    )

    catexps = list(catexps_hsc.values()) + [catexp_hst]
    return catexps, catalog_ref_hsc, catalog_hst, config_data, fitter


class CosmosHstTable(ObjectTableBase):
    """Class for retrieving columns from a COSMOS catalog."""

    wcs: SkyWcs = pydantic.Field(doc="The object table")

    @cached_property
    def _flux(self):
        return 10 ** (-0.4 * (self.table[f"mag_best"] - 23.9))

    @cached_property
    def _xy(self):
        return self.wcs.skyToPixelArray(self.table["ra"]*np.pi/180, self.table["dec"]*np.pi/180)

    def get_flux(self, band: str) -> np.ndarray:
        return self._flux

    def get_id(self) -> np.ndarray:
        return self.table["index_row"]

    def get_is_extended(self) -> np.ndarray:
        return self.table["mu_class"] == 1  # noqa: E712

    def get_is_variable(self) -> np.ndarray:
        return self.table["mu_class"] == 2  # noqa: E712

    def get_x(self):
        return self._xy[0]

    def get_y(self):
        return self._xy[1]
