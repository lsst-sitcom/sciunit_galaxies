import lsst.daf.butler as dafButler
from lsst.analysis.tools.atools.diffMatched import *
from lsst.analysis.tools.actions.vector.selectors import InjectedGalaxySelector, InjectedStarSelector
from lsst.analysis.tools.atools import SizeMagnitudePlot
from lsst.analysis.tools.contexts import CoaddContext
import matplotlib as mpl

import os

skymap = "lsst_cells_v1"
tract = 5063

InjectedGalaxySelector.key_class.default = "ref_r_comp1_source_type"
InjectedStarSelector.key_class.default = "ref_r_comp1_source_type"

collection = "u/dtaranu/DM-50425/injected_dp1_v29_0_0_rc6/plots"
butler = dafButler.Butler("/repo/main", collections=collection)

objects = butler.get("object_all", skymap=skymap, tract=tract, storageClass="ArrowAstropy")
plotInfo = {
    "run": collection,
    "tract": tract,
    "skymap": skymap,
}
mmag_min, mmag_max = 17500, 27500
mag_min, mag_max = mmag_min/1000, mmag_max/1000.
kwargs_produce = {"xLims": (mag_min, mag_max), }
kwargs_produce_chi = kwargs_produce.copy()
kwargs_produce_chi.update({"yLims": (-9, 9)})

# stars otherwise
for do_galaxies in (False, True):
    suffix_folder = "_galaxies" if do_galaxies else "_stars"
    output_dir = f"dp1_paper_plots{suffix_folder}"

    reconfigure_diff_matched_defaults(
        config=None,
        context="injection",
        key_flux_meas="cmodel_err",
        use_any=False,
        use_galaxies=do_galaxies,
        use_stars=not do_galaxies,
    )

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
                "produce": {"xLims": (mag_min, mag_max), "yLims": (-4, 3),},
                "applyContext": CoaddContext,
            }),
        },
        "matched_injected_deep_coadd_predetection_catalog_tract_injected_object_all": {
            "completeness": (
                MatchedRefCoaddCompurityTool,
                {
                    "mag_bins_plot": {"mag_low_min": mmag_min, "mag_low_max": mmag_max},
                    "produce": {
                        "label_shift": -0.15,
                        "legendLocation": "outside upper center",
                        "mag_ref_label": "{band}-band Injected Magnitude",
                        "reference_label": "Injected",
                        "show_purity": False,
                    },
                    "reconfigure": {"use_any": do_galaxies, "use_galaxies": False, "use_stars": not do_galaxies},
                }
            ),
            "ra": (MatchedRefCoaddDiffCoordRaTool, {"produce": kwargs_produce}),
            "dec": (MatchedRefCoaddDiffCoordDecTool, {"produce": kwargs_produce}),
            "mag_cmodel": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce}),
            "mag_kron": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce, "mag_y": "kron_err"}),
            "mag_psf": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce, "mag_y": "psf_err"}),
            "mag_sersic": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce, "mag_y": "sersic_err"}),
            "mag_chi_cmodel": (MatchedRefCoaddChiMagTool, {"produce": kwargs_produce_chi}),
            "mag_chi_psf": (MatchedRefCoaddChiMagTool, {"produce": kwargs_produce_chi, "mag_y": "psf_err"}),
            "mag_chi_sersic": (MatchedRefCoaddChiMagTool, {"produce": kwargs_produce_chi, "mag_y": "sersic_err"}),
            "color_cmodel": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce}),
            "color_gaap": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "gaap1p0_err"}),
            "color_psf": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "psf_err"}),
            "color_sersic": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "sersic_err"}),
            "color_chi_cmodel": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi}),
            "color_chi_gaap": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "gaap1p0_err"}),
            "color_chi_psf": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "psf_err"}),
            "color_chi_sersic": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "sersic_err"}),
            "sersic_ra": (MatchedRefCoaddDiffCoordRaTool, {"produce": kwargs_produce}),
            "sersic_dec": (MatchedRefCoaddDiffCoordDecTool, {"produce": kwargs_produce}),
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
