import logging
import os
import pathlib

from lsst.multiprofit import SersicComponentConfig
from lsst.sitcom.sciunit.galaxies.fit_cosmos_hst_objects import MakeCachedCosmosHstInitializerAction
from lsst.sitcom.sciunit.galaxies.read_cosmos_data import build_fit_inputs, get_deblended_model_fits_filepath

testdata_cosmos_dir = os.environ["TESTDATA_COSMOS_DIR"]
tract: int = 9813
patch: int = 40

bands_hsc = ["g", "r", "i"]
band_hsc_ref = "i"
band_hst = "F814W"
chained = True
chromatic_centroid = True
fit_exponential = False
save = True
suffix = "_achrom" if not chromatic_centroid else ""
suffix = f"{suffix}{'_chained' if chained else ''}"

config_sersic = SersicComponentConfig(
    prior_axrat_mean=0.7,
    prior_axrat_stddev=0.2,
    prior_size_stddev=0.1,
)
action_initializer = MakeCachedCosmosHstInitializerAction()
action_initializer.config.rho_abs_max = 0.9
name_model = "exponential" if fit_exponential else "sersic"
kwargs_initializer = {}
if fit_exponential:
    config_sersic.sersic_index.fixed = True
    config_sersic.sersic_index.value_initial = 1.0
elif chained:
    from lsst.daf.butler.formatters.parquet import arrow_to_astropy, pq
    from lsst.meas.extensions.multiprofit.input_config import InputConfig

    action_initializer.config_cosmos.chained = True
    filename = get_deblended_model_fits_filepath(model="exponential", tract=tract, patch=patch)
    tab_init = arrow_to_astropy(pq.read_table(f"{testdata_cosmos_dir}/{filename}"))
    input_config = InputConfig(
        doc="Exponential fit results",
        is_multiband=True,
    )
    kwargs_initializer["exponential"] = (input_config, tab_init)

configs_sersic: dict[str, SersicComponentConfig] = {name_model: config_sersic}

logging.basicConfig(level=logging.INFO)

catexps, catalog_ref_hsc, catalog_hst, config_data, fitter = build_fit_inputs(
    testdata_cosmos_dir=testdata_cosmos_dir,
    bands_hsc=bands_hsc,
    tract=tract, patch=patch,
    band_hsc_ref=band_hsc_ref,
    chromatic_centroid=chromatic_centroid,
    configs_sersic=configs_sersic,
    action_initializer=action_initializer,
    kwargs_initializer=kwargs_initializer,
)
# Try to get the centroids to move more
# config_data.config.centroid_scale_factor = 0.1

if save:
    filename = get_deblended_model_fits_filepath(
        model=f"{name_model}{suffix}", tract=tract, patch=patch,
    )
    filepath = f"{testdata_cosmos_dir}/{filename}"
    folder = pathlib.Path(filepath).parent
    if os.path.exists(folder):
        if os.path.isfile(folder):
            raise RuntimeError(f"{folder} is a file")
    else:
        os.makedirs(folder)

results = fitter.fit(
    catalog_multi=catalog_ref_hsc,
    catexps=catexps,
    config_data=config_data,
)

if save:
    from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size, pq
    tab_arrow = astropy_to_arrow(results)
    pq.write_table(tab_arrow, filepath, row_group_size=compute_row_group_size(tab_arrow.schema))
