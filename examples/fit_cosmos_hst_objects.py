import logging
import os

from lsst.sitcom.sciunit.galaxies.read_cosmos_data import build_fit_inputs, get_deblended_model_fits_filepath
import pyarrow.parquet as pq

testdata_cosmos_dir = os.environ["TESTDATA_COSMOS_DIR"]
tract: int = 9813
patch: int = 40

bands_hsc = ["g", "r", "i"]
band_hsc_ref = "i"
band_hst = "F814W"

logging.basicConfig(level=logging.INFO)

catexps, catalog_ref_hsc, catalog_hst, config_data, fitter = build_fit_inputs(
    testdata_cosmos_dir=testdata_cosmos_dir,
    bands_hsc=bands_hsc,
    tract=tract, patch=patch,
    band_hsc_ref=band_hsc_ref,
)

results = fitter.fit(
    catalog_multi=catalog_ref_hsc,
    catexps=catexps,
    config_data=config_data,
)

save = True
if save:
    from lsst.daf.butler.formatters.parquet import astropy_to_arrow
    filename = get_deblended_model_fits_filepath(model='sersic', tract=tract, patch=patch)
    pq.write_table(astropy_to_arrow(results), f"{testdata_cosmos_dir}/{filename}")
