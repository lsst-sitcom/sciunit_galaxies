import logging
import os

from lsst.daf.butler.formatters.parquet import arrow_to_astropy, pq
from lsst.meas.extensions.multiprofit.fit_coadd_multiband import MultiProFitSourceTask
from lsst.meas.extensions.multiprofit.plots import bands_weights_lsst, plot_blend
from lsst.meas.extensions.multiprofit.rebuild_coadd_multiband import (
    ModelRebuilder, PatchCoaddRebuilder, PatchModelMatches,
)
from lsst.multiprofit.plotting.plot_model_singleband import plot_model_singleband
from lsst.sitcom.sciunit.galaxies.read_cosmos_data import (
    build_fit_inputs, CosmosHstTable, get_dataset_filename_first, get_deblended_model_fits_filepath
)
import numpy as np

tract: int = 9813
patch: int = 40

bands_hsc = ["g", "r", "i"]
band_hsc_ref = "i"
band_hst = "F814W"
write_cutouts = False
show_plots = True

testdata_cosmos_dir = os.environ["TESTDATA_COSMOS_DIR"]
path_cutouts = f"{testdata_cosmos_dir}/cutouts"
prefix_path = f"{testdata_cosmos_dir}/"

logging.basicConfig(level=logging.INFO)

fit_obj = arrow_to_astropy(pq.read_table(
    f"{prefix_path}{get_deblended_model_fits_filepath(tract=tract, patch=patch)}"
))
objects = arrow_to_astropy(pq.read_table(get_dataset_filename_first(
    prefix_path=prefix_path, dataset="objectTable_tract", tract=tract, patch=None, suffix="*.parq",
)))

catexps, catalog_ref_hsc, catalog_hst, config_data, fitter = build_fit_inputs(
    testdata_cosmos_dir=testdata_cosmos_dir,
    bands_hsc=bands_hsc,
    tract=tract, patch=patch,
    band_hsc_ref=band_hsc_ref,
)

task_fit = MultiProFitSourceTask(config=config_data.config)

rebuilder_model = ModelRebuilder(
    catexps=catexps,
    catalog_multi=catalog_ref_hsc,
    fit_results=fit_obj,
    fitter=fitter,
    task_fit=task_fit,
)

rebuilder = PatchCoaddRebuilder(
    matches={"sersic": PatchModelMatches(matches=None, quantumgraph=None, rebuilder=rebuilder_model)},
    name_model_ref="sersic",
    objects=objects, objects_multiprofit=fit_obj,
    reference=catalog_hst, skymap="hsc_rings_v1", tract=tract, patch=patch,
)

catexp_ref = next(iter(catexp for catexp in catexps if catexp.channel.name == band_hsc_ref))
wcs = catexp_ref.exposure.wcs

mag_hst = -2.5*np.log10(fit_obj["mpf_sersic_F814W_flux"]) + 31.4 - 7.5
idx_bright = np.where(mag_hst < 21.5)[0].astype(np.int32)
weights = {band: bands_weights_lsst[band] for band in reversed(bands_hsc)}

for idx in idx_bright:
    if write_cutouts:
        for band in bands_hsc:
            catalog, exposure = catexps[band]
            bbox = catalog[idx].getFootprint().getBBox()
            cutout = exposure.getCutout(bbox)
            img_calibrated = exposure.photoCalib.calibrateImage(exposure.maskedImage[bbox])
            cutout.image.array = img_calibrated.image.array
            cutout.variance.array = img_calibrated.variance.array
            cutout

    plot_blend(
        rebuilder, idx_row_parent=idx, weights=weights,
        table_ref_type=CosmosHstTable,
        kwargs_table_ref={"wcs": wcs},
    )
    model = rebuilder_model.fitter.get_model(
        idx,
        catalog_multi=rebuilder_model.catalog_multi,
        catexps=rebuilder_model.catexps,
        config_data=rebuilder_model.config_data,
        results=rebuilder_model.fit_results,
    )
    plot_model_singleband(model=model, idx_obs=3)
