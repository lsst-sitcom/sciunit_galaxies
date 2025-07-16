import lsst.daf.butler as dafButler
from lsst.analysis.tools.atools.diffMatched import *
from lsst.analysis.tools.actions.vector.selectors import InjectedGalaxySelector, InjectedStarSelector
from lsst.analysis.tools.atools import SizeMagnitudePlot
from lsst.analysis.tools.contexts import CoaddContext
import matplotlib as mpl

import os

output_dir = "dp1_paper_plots"

skymap = "lsst_cells_v1"
tract = 5063

InjectedGalaxySelector.key_class.default = "ref_r_comp1_source_type"
InjectedStarSelector.key_class.default = "ref_r_comp1_source_type"

reconfigure_diff_matched_defaults(
    config=None,
    context="injection",
    key_flux_meas="cmodel_err",
    use_any=False,
    use_galaxies=True,
    use_stars=False,
)

collection = "u/dtaranu/DM-50425/injected_dp1_v29_0_0_rc6/plots"

butler = dafButler.Butler("/repo/main", collections=collection)

objects = butler.get("object_all", skymap=skymap, tract=tract, storageClass="ArrowAstropy")
plotInfo = {
    "run": collection,
    "tract": tract,
    "skymap": skymap,
}
dataset_tools = {
    "object_all": {
        "sersic_size": (SizeMagnitudePlot, {
            "config_moments": {
                "xx": "reff_x",
                "yy": "reff_y",
                "xy": "rho",
            },
            "is_covariance": False,
            "mag_x": "sersic_err",
            "size_type": "determinantRadius",
            "size_y": "sersic",
            "produce": {"xLims": (16.5, 29), "yLims": (-4, 3),},
            "applyContext": CoaddContext,
        }),
    },
    "matched_injected_deep_coadd_predetection_catalog_tract_injected_object_all": {
        "completeness": (
            MatchedRefCoaddCompurityTool,
            {
                "mag_bins_plot": {"mag_low_min": 16500, "mag_low_max": 29000},
                "produce": {"label_shift": -0.15, "legendLocation": "outside upper center", "show_purity": False},
                "reconfigure": {"use_any": True, "use_galaxies": False,},
            }
        ),
        "ra": (MatchedRefCoaddDiffCoordRaTool, {}),
        "dec": (MatchedRefCoaddDiffCoordDecTool, {}),
        "mag_cmodel": (MatchedRefCoaddDiffMagTool, {}),
        "mag_kron": (MatchedRefCoaddDiffMagTool, {"mag_y": "kron_err"}),
        "mag_psf": (MatchedRefCoaddDiffMagTool, {"mag_y": "psf_err"}),
        "mag_sersic": (MatchedRefCoaddDiffMagTool, {"mag_y": "sersic_err"}),
        "mag_chi_cmodel": (MatchedRefCoaddChiMagTool, {}),
        "color_cmodel": (MatchedRefCoaddDiffColorTool, {}),
        "color_gaap": (MatchedRefCoaddDiffColorTool, {"mag_y1": "gaap1p0_err"}),
        "color_psf": (MatchedRefCoaddDiffColorTool, {"mag_y1": "psf_err"}),
        "color_sersic": (MatchedRefCoaddDiffColorTool, {"mag_y1": "sersic_err"}),
        "sersic_ra": (MatchedRefCoaddDiffCoordRaTool, {}),
        "sersic_dec": (MatchedRefCoaddDiffCoordDecTool, {}),
    },
}
bands = ("i", "r")

if not os.path.exists(output_dir):
    print(f"{output_dir=} does not exist; making it now")
    os.mkdir(output_dir)
elif not os.path.isdir(output_dir):
    raise RuntimeError(f"{output_dir=} exists but is not a directory")


def apply_override(atool, attr, value):
    if isinstance(value, dict):
        atool_attr = getattr(atool, attr)
        for k, v in value.items():
            apply_override(atool_attr, k, v)
    else:
        setattr(atool, attr, value)


for dataset, tools in dataset_tools.items():
    plotInfo["tableName"] = dataset
    data = butler.get(dataset, skymap=skymap, tract=tract, storageClass="ArrowAstropy")
    for name, (class_tool, overrides) in tools.items():
        atool = class_tool()
        overrides_produce = overrides.pop("produce", {})
        overrides_reconfigure = overrides.pop("reconfigure", {})
        if overrides_reconfigure:
            atool.reconfigure(**overrides_reconfigure)
        for attr, value in overrides.items():
            apply_override(atool, attr, value)
        atool.finalize()
        produce_plot = atool.produce.plot
        plots = produce_plot.actions if hasattr(produce_plot, "actions") else [produce_plot]
        for plot in plots:
            plot.publicationStyle = True
            for attr, value in overrides_produce.items():
                apply_override(plot, attr, value)
        for band in bands:
            plotInfo["bands"] = [band]
            results = atool(data, band=band, plotInfo=plotInfo, skymap=skymap)
            for name_plot, result in results.items():
                if isinstance(result, mpl.figure.Figure):
                    suffix = "" if (not "_" in name_plot) else f'_{name_plot.rsplit("_", 1)[0]}'
                    result.savefig(f"{output_dir}/injected_{skymap}_{tract}_{band}_{name}{suffix}.pdf")
