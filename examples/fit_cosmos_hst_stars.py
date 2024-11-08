import logging
import os

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from lsst.daf.butler.formatters.parquet import arrow_to_astropy
import lsst.gauss2d.fit as g2f
from lsst.multiprofit import ComponentGroupConfig, GaussianComponentConfig, ModelConfig, SourceConfig
from lsst.multiprofit.fitting import CatalogSourceFitterConfigData
from lsst.sitcom.sciunit.galaxies.fit_cosmos_hst import CatalogExposureCosmosHstBase
from lsst.sitcom.sciunit.galaxies.fit_cosmos_hst_stars import (
    CatalogExposureCosmosHstStars, MultiProFitCosmosHstStarsConfig, MultiProFitCosmosHstStarsFitter,
)
from lsst.sitcom.sciunit.galaxies.read_cosmos_data import (
    get_psf_model_fits_filepath, get_tiles_from_patch, read_data_hsc,
)
import numpy as np
import pyarrow.parquet as pq

testdata_cosmos_dir = os.environ["TESTDATA_COSMOS_DIR"]
tract: int = 9813
patch: int = 40

bands_hsc = {"g", "r", "i"}

logging.basicConfig(level=logging.INFO)

_, catalog_ref_hsc = read_data_hsc(testdata_cosmos_dir, bands=[])
ra_hsc, dec_hsc = (catalog_ref_hsc[col] for col in ("coord_ra", "coord_dec"))
radec_min, radec_max = tuple(
    SkyCoord(*[
        (f(coord)*u.rad).to(u.deg) for coord in (ra_hsc, dec_hsc)
    ])
    for f in (np.min, np.max)
)

tile_table = Table.read(f"{testdata_cosmos_dir}/cosmos_hst_tiles.ecsv")
path_tiles = f"{testdata_cosmos_dir}/cosmos_hst_tiles/"
tiles_within = get_tiles_from_patch(
    tile_table=tile_table, radec_min=radec_min, radec_max=radec_max, path_tiles=path_tiles,
)

band_hst = "F814W"
catalog_hst = arrow_to_astropy(pq.read_table(f"{testdata_cosmos_dir}/cosmos_acs_iphot_200709.parq"))
catalog_hst["index_row"] = np.arange(len(catalog_hst))
within_catalog = (catalog_hst["ra"] > radec_min.ra) & (catalog_hst["ra"] < radec_max.ra) & (
    catalog_hst["dec"] > radec_min.dec) & (catalog_hst["dec"] < radec_max.dec)
catalog_hst = catalog_hst[within_catalog]

fitter = MultiProFitCosmosHstStarsFitter()

obs_hst, wcs_hst, cos_dec_hst = CatalogExposureCosmosHstBase.get_observation_hst(
    tiles_within, radec_min=radec_min, radec_max=radec_max, band=band_hst,
)

n_gaussians = 3

components_gauss = {
    str(idx + 1): GaussianComponentConfig(prior_axrat_mean=0.95, prior_axrat_stddev=0.1)
    for idx in range(n_gaussians)
}

config_fit = MultiProFitCosmosHstStarsConfig(
    config_model=ModelConfig(
        sources={
            "": SourceConfig(
                component_groups={
                    "": ComponentGroupConfig(components_gauss=components_gauss)
                }
            )
        }
    ),
    flag_errors={v: str(k) for k, v in fitter.errors_expected.items()},
)
catexp_hst = CatalogExposureCosmosHstStars(
    catalog_hst=catalog_hst,
    catalog_ref_hsc=catalog_ref_hsc,
    config_fit=config_fit,
    cos_dec_hst=cos_dec_hst,
    dataId={"tract": tract, "patch": patch, "band": band_hst},
    observation_hst=obs_hst,
    wcs_hst=wcs_hst,
)
config_data = CatalogSourceFitterConfigData(config=config_fit, channels=[catexp_hst.channel])

results = fitter.fit(catalog_multi=catalog_ref_hsc, catexps=[catexp_hst], config_data=config_data)
CatalogExposureCosmosHstStars.populate_psf_table(results, catalog_ref_hsc, band=band_hst)

save = True
if save:
    from lsst.daf.butler.formatters.parquet import astropy_to_arrow
    filepath = get_psf_model_fits_filepath(tract=tract, patch=patch, band=band_hst)
    pq.write_table(astropy_to_arrow(results), f"{testdata_cosmos_dir}/{filepath}")
