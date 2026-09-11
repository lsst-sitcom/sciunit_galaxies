from lsst.ts.xml import field_info

from lsst.analysis.tools.interfaces import AnalysisTool
from lsst.analysis.tools.atools.diffMatched import *
from lsst.analysis.tools.atools.genericBuild import FluxConfig
from lsst.analysis.tools.actions.vector import (
    CompositeSelector,
    FlagSelector,
    GalaxySelector,
    LoadVector,
    MatchedTractSelector,
    RangeSelector,
    ReferenceGalaxySelector,
    ReferenceObjectSelector,
    ReferenceStarSelector,
    SetSelector,
    StarSelector,
    ThresholdSelector,
)
from lsst.analysis.tools.atools import SizeMagnitudePlot
from lsst.analysis.tools.actions.plot.patchActionSkyPlot import PerPatchMetricConfig, PerPatchPropertyMapPlot
import lsst.daf.butler as dafButler
from lsst.geom import degrees, SpherePoint
from lsst.sitcom.sciunit.galaxies.ddfs.dp2 import CosmosInfo, EcdfsInfo, EdfsInfo
#from lsst.sitcom.sciunit.galaxies.plotting import PerPatchMetricConfig, PerPatchPropertyMapPlot

from astropy.table import vstack
import matplotlib as mpl
import numpy as np

import argparse
import logging
import os

field_infos = {
    "cosmos": CosmosInfo,
    "ecdfs": EcdfsInfo,
    "edfs": EdfsInfo,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("make_dp2_paper_plots")
    parser.add_argument("--completeness_only", help="Only make completeness plots", action="store_true")
    parser.add_argument("--dataset", help="Matched dataset type name", type=str, default=None)
    parser.add_argument("--dec_min", help="Minimum dec for plots", type=float, default=None)
    parser.add_argument("--dec_max", help="Maximum dec for plots", type=float, default=None)
    parser.add_argument("--log_level", help="Logging level", type=str, default="INFO")
    parser.add_argument("--field", help="Name of the field", type=str, default="ecdfs")
    parser.add_argument("--interactive", help="Make and show plots interactively", action="store_true")
    parser.add_argument("--name_release", help="Name of the release", type=str, default="DP2")
    parser.add_argument("--ra_min", help="Minimum RA for plots", type=float, default=None)
    parser.add_argument("--ra_max", help="Maximum RA for plots", type=float, default=None)
    parser.add_argument("--repo", help="Butler repo to load skymap from", type=str, default="dp2_prep")
    parser.add_argument("--skymap_name", help="Name of the skymap", type=str, default="lsst_cells_v2")
    parser.add_argument("--weekly", help="Pipelines version", type=str, default="v30_0_8")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    mpl.rcParams.update({"image.origin": "lower", "font.size": 10, "figure.figsize": (8, 8)})

    skymap = args.skymap_name
    field = args.field
    field_info = field_infos[field]
    name_release = args.name_release
    prefix_release = f"{name_release}_" if name_release else ""
    weekly = args.weekly
    collection = f"u/dtaranu/DM-50135/{name_release}{'/' if name_release else ''}{weekly}/matched_{field}"

    metrics_plot = {
        "mag_compl50": PerPatchMetricConfig(
            description="{band} mag @50% completeness",
            key="{band}_detect_{name_flux_target}_vs_{name_flux_ref}_{suffix_metric}_mag_completeness_50p00_pct",
            vmin=25.5,
            vmax=26.5,
        ),
        "mag_compl80": PerPatchMetricConfig(
            description="{band} mag @80% completeness",
            key="{band}_detect_{name_flux_target}_vs_{name_flux_ref}_{suffix_metric}_mag_completeness_80p00_pct",
            vmin=25,
            vmax=26,
        ),
        "mag_compl90": PerPatchMetricConfig(
            description="{band} mag @90% completeness",
            key="{band}_detect_{name_flux_target}_vs_{name_flux_ref}_{suffix_metric}_mag_completeness_90p00_pct",
            vmin=24,
            vmax=25,
        ),
        "compl_24_25": PerPatchMetricConfig(
            description="24<{band}<25 completeness",
            key="{band}_detect_{name_flux_target}_vs_{name_flux_ref}_{suffix_metric}_completeness_mag24p0",
            vmin=0.80,
            vmax=0.95,
        ),
    }

    ra_min = args.ra_min
    ra_max = args.ra_max
    dec_min = args.dec_min
    dec_max = args.dec_max

    metrics_plot_band = {}

    ref_name = args.dataset or field_info.refcat_names[0]
    tract_patches = field_info.tract_patches
    dataset_type = f"matched_{ref_name}_object"
    tracts = tuple(tract_patches.keys())

    butler = dafButler.Butler("dp2_prep", collections=collection)
    skymapInfo = butler.get("skyMap", skymap=skymap)

    plotInfo = {
        "run": collection,
        "tract": ",".join(str(tract) for tract in tracts),
        "skymap": skymap,
    }
    mmag_min, mmag_max = 17500, 27500
    mag_min, mag_max = mmag_min/1000, mmag_max/1000.
    kwargs_produce = {"xLims": (mag_min, mag_max), }
    kwargs_produce_chi = kwargs_produce.copy()
    kwargs_produce_chi.update({"yLims": (-9, 9)})

    if ref_name == "euclid_q1":
        weights_lsst_vis = {"g": 0.5, "r": 1.0, "i": 1.0, "z": 1.0}
        bands_lsst = tuple(weights_lsst_vis.keys()) + ("y",)
        bands = ("g", "r", "i", "vis", "z", "y")
        bands_ref = ("vis", "y")
        bands_ref_matched = {"vis": ("g", "r", "i", "vis", "z"), "y": ("y",)}
        bands_completeness = bands_ref
        bands_color = {"vis": "y"}
        refcat_ra = "refcat_right_ascension"
        refcat_dec = "refcat_declination"

        selector_obj_ref = ReferenceObjectSelector(vectorKey="refcat_spurious_prob", minimum=0, maximum=0.1)
        selector_galaxy_ref, selector_star_ref = (CompositeSelector(combine_by_and=True) for _ in range(2))
        for selector in selector_galaxy_ref, selector_star_ref:
            selector.selectors.spurious = selector_obj_ref
        selector_galaxy_ref.selectors.galaxy = ReferenceGalaxySelector(
            plotLabelValue="Euclid mumax_minus_mag >= -2.6",
            vectorKey="refcat_mumax_minus_mag", threshold=-2.6, op="ge",
        )
        selector_star_ref.selectors.star = ReferenceStarSelector(
            plotLabelValue="Euclid mumax_minus_mag < -2.6",
            vectorKey="refcat_mumax_minus_mag",
            threshold=-2.6,
            op="le",
        )
        ref_matched = FluxConfig(
            key_flux="refcat_flux_{band}_sersic",
            name_flux="{band} Sersic",
            name_flux_short="sersic",
            key_flux_error=None,
        )
        tract_keys = ["tract", "refcat_tract"]
        patch_keys = ["patch", "refcat_patch"]
    elif ref_name == "des_y6gold":
        weights_lsst_vis = {"g": 1.0, "r": 1.0, "i": 1.0, "z": 1.0}
        bands_lsst = ("g", "r", "i", "z", "y")
        bands = bands_lsst
        bands_ref = bands_lsst
        bands_ref_matched = {}
        bands_completeness = bands_lsst
        bands_color = {"g": "r,i", "r": "i", "i": "z", "z": "y"}
        refcat_ra = "refcat_alphawin_j2000"
        refcat_dec = "refcat_deltawin_j2000"

        selector_galaxy_ref = ReferenceGalaxySelector(
            plotLabelValue="DES ext_mash > 1.5",
            vectorKey="refcat_ext_mash", threshold=1.5, op="ge",
        )
        selector_obj_ref = ReferenceObjectSelector(vectorKey="refcat_ext_mash", minimum=0)
        selector_star_ref = RangeSelector(
            plotLabelKey="DES 0 < ext_mash < 1.5",
            vectorKey="refcat_ext_mash",
            minimum=0, maximum=1.5,
        )
        ref_matched = FluxConfig(
            key_flux="refcat_bdf_flux_{band}_corrected",
            name_flux="{band} BDF",
            name_flux_short="BDF",
            key_flux_error=None,
        )
        tract_keys = ["tract", "refcat_tract"]
        patch_keys = ["patch", "refcat_patch"]
    elif ref_name == "cosmos_acs_iphot_200709":
        weights_lsst_vis = {"i": 1.0, "z": 1.0}
        bands_lsst = ("g", "r", "i", "z", "y")
        bands = ("vis", "g", "r", "i", "z", "y")
        bands_ref = ("F814W",)
        bands_ref_matched = {}
        bands_completeness = ("vis",)
        bands_color = {"g": "r,i", "r": "i", "i": "z", "z": "y"}
        refcat_ra = "refcat_ra"
        refcat_dec = "refcat_dec"

        selector_galaxy_ref = ReferenceGalaxySelector(
            plotLabelKey="Selection: HST galaxies", vectorKey="refcat_mu_class", threshold=1,
        )
        selector_obj_ref = ReferenceObjectSelector(vectorKey="refcat_mu_class", maximum=3)
        selector_star_ref = ReferenceStarSelector(
            plotLabelKey="Selection: HST stars", vectorKey="refcat_mu_class", threshold=2,
        )
        ref_matched = FluxConfig(
            key_flux="refcat_flux_auto",
            name_flux="F814W Auto",
            name_flux_short="F814W",
            key_flux_error=None,
        )
        tract_keys = []
        patch_keys = ["patch"]
    else:
        raise ValueError(f"Unknown {ref_name=}")

    if args.interactive:
        import matplotlib.pyplot as plt

    bands_ref_matched_default = {None: ("default",)}

    if not bands_ref_matched:
        bands_ref_matched = bands_ref_matched_default

    sum_weights_lsst_vis = sum(weights_lsst_vis.values())

    reconfigure_color = {"bands_color": bands_color}

    selector_all = SetSelector(vectorKeys=["match_candidate", "refcat_match_candidate"], values=[1])
    selector_galaxy = GalaxySelector(
        vectorKey="griz_model_extendedness", extendedness_minimum=0.2,
    )
    selector_star = StarSelector(
        vectorKey="griz_model_extendedness", extendedness_maximum=0.2,
    )

    MatchedRefCoaddDiffMagTool.fluxes_default.ref_matched = ref_matched
    MatchedRefCoaddCompurityTool.fluxes_default.ref_matched = ref_matched
    # action_completeness_plot.selector_star = selector_star_ref
    # action_completeness_plot.key_flux_ref = ref_matched.key_flux

    selector_candidate = CompositeSelector()
    selector_candidate.selectors.candidate = SetSelector(
        vectorKeys=("match_candidate", "refcat_match_candidate"),
        values=[1],
    )
    if tract_keys:
        selector_tract_patches = CompositeSelector(combine_by_and=False)
        for tract, patches in tract_patches.items():
            selector_tract = MatchedTractSelector(tract=int(tract), patches=patches) if patches else (
                SetSelector(vectorKeys=tract_keys, values=[tract]))
            setattr(selector_tract_patches.selectors, f"tract_{tract}", selector_tract)
        selector_candidate.selectors.tract_patches = selector_tract_patches
        selector_tracts = SetSelector(vectorKeys=tract_keys, values=list(tract_patches.keys()))
    else:
        assert len(tract_patches) == 1
        tract, patches = next(iter(tract_patches.items()))
        if patches:
            selector_candidate.selectors.tract_patches = SetSelector(vectorKeys=["patch"], values=patches)
        selector_tracts = None

    dataset_tools = {
        "object": {
            "sersic_size": (SizeMagnitudePlot, {
                "mag_x": "sersic_err",
                "size_type": "singleColumnSize",
                "size_y": "sersic",
                "produce": {"xLims": (mag_min, mag_max), "yLims": (-3, 2),},
                "selector_all": ThresholdSelector(op="ge", vectorKey="griz_model_extendedness", threshold=0),
                # "applyContext": CoaddContext,
                "prep": {
                    "selectors": {"flagSelector": FlagSelector(selectWhenFalse=[
                        "{band}_pixelFlags_saturatedCenter",
                        "coord_flag",
                    ])},
                },
            }),
        },
        dataset_type: {
            "completeness": (
                MatchedRefCoaddCompurityTool,
                {
                    "config_metrics": {"completeness_mag_max": 20},
                    "mag_bins_plot": {"mag_low_min": mmag_min, "mag_low_max": mmag_max},
                    "make_patch_sky_plots": True,
                    "produce": {
                        "default": {
                            "label_shift": -0.25,
                            "legendLocation": "outside upper center",
                            "show_purity": False,
                        },
                        "patch_sky_plots": {},
                    },
                }
            ),
            "ra": (
                MatchedRefCoaddDiffCoordRaTool,
                {"produce": kwargs_produce, "coord_ref": refcat_ra, "coord_ref_cos": refcat_dec},
            ),
            "dec": (
                MatchedRefCoaddDiffCoordDecTool,
                {"produce": kwargs_produce, "coord_ref": refcat_dec},
            ),
            "mag_cmodel": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce}),
            "mag_kron": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce, "mag_y": "kron_err"}),
            "mag_psf": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce, "mag_y": "psf_err"}),
            "mag_sersic": (MatchedRefCoaddDiffMagTool, {"produce": kwargs_produce, "mag_y": "sersic_err"}),
            "mag_chi_cmodel": (MatchedRefCoaddChiMagTool, {"produce": kwargs_produce_chi}),
            "mag_chi_psf": (MatchedRefCoaddChiMagTool, {"produce": kwargs_produce_chi, "mag_y": "psf_err"}),
            "mag_chi_sersic": (MatchedRefCoaddChiMagTool, {"produce": kwargs_produce_chi, "mag_y": "sersic_err"}),
            "color_cmodel": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "reconfigure": reconfigure_color}),
            "color_gaap": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "gaap1p0_err", "reconfigure": reconfigure_color}),
            "color_psf": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "psf_err", "reconfigure": reconfigure_color}),
            "color_sersic": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "sersic_err", "reconfigure": reconfigure_color}),
            "color_chi_cmodel": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "reconfigure": reconfigure_color}),
            "color_chi_gaap": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "gaap1p0_err", "reconfigure": reconfigure_color}),
            "color_chi_psf": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "psf_err", "reconfigure": reconfigure_color}),
            "color_chi_sersic": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "sersic_err", "reconfigure": reconfigure_color}),
            "sersic_ra": (
                MatchedRefCoaddDiffCoordRaTool,
                {"produce": kwargs_produce, "coord_ref": refcat_ra, "coord_ref_cos": refcat_dec, "coord_meas": "sersic_ra"},
            ),
            "sersic_dec": (
                MatchedRefCoaddDiffCoordDecTool,
                {"produce": kwargs_produce, "coord_ref": refcat_dec, "coord_meas": "sersic_dec"},
            ),
        },
    }

    overrides_all = {
        dataset_type: {
            "prep": {
                "selectors": {"match_candidate": selector_candidate},
            },
            "selector_all": selector_all,
            "selector_galaxy": selector_galaxy,
            "selector_star": selector_star,
            "selector_ref_galaxy": selector_galaxy_ref,
            "selector_ref_all": selector_obj_ref,
            "selector_ref_star": selector_star_ref,
        }
    }

    if args.completeness_only:
        dataset_tools[dataset_type] = {
            "completeness": dataset_tools[dataset_type]["completeness"]
        }
        del dataset_tools["object"]

    def apply_override(atool, attr, value):
        if isinstance(value, dict):
            atool_attr = getattr(atool, attr)
            for k, v in value.items():
                apply_override(atool_attr, k, v)
        else:
            setattr(atool, attr, value)

    for dataset, tools in dataset_tools.items():
        tools_object_type = {}
        columns = {f"{band}_psfFlux" for band in bands_lsst}
        is_matched = dataset.startswith("matched_")

        # stars otherwise
        for do_galaxies_only in (True, False, None):
            do_galaxies = do_galaxies_only == True
            suffix_folder = "all" if (do_galaxies_only is None) else ("galaxies" if do_galaxies else "stars")
            suffix_metric = "all" if (do_galaxies_only is None) else (
                "resolved" if do_galaxies else "unresolved")
            output_dir = f"dp2_{field}_plots_{prefix_release}{weekly}_{suffix_folder}"
            tools_named = {}

            if not os.path.exists(output_dir):
                print(f"{output_dir=} does not exist; making it now")
                os.mkdir(output_dir)
            elif not os.path.isdir(output_dir):
                raise RuntimeError(f"{output_dir=} exists but is not a directory")

            if is_matched:
                reconfigure_diff_matched_defaults(
                    config=None,
                    context="custom",
                    key_flux_meas="sersic_err",
                    use_any=do_galaxies_only is None,
                    use_galaxies=do_galaxies,
                    use_stars=not do_galaxies_only,
                )

            plotInfo["tableName"] = dataset
            for name, (class_tool, overrides_tool) in tools.items():
                if (overrides := overrides_all.get(dataset)) is not None:
                    overrides = overrides.copy()
                    overrides.update(overrides_tool)
                else:
                    overrides = overrides_tool
                atool = class_tool()
                overrides_produce = overrides.pop("produce", {})
                overrides_reconfigure = overrides.pop("reconfigure", {})
                if name == "completeness":
                    overrides_reconfigure["use_any"] = do_galaxies_only is None
                    overrides_reconfigure["use_galaxies"] = do_galaxies
                    overrides_reconfigure["use_stars"] = not do_galaxies_only
                if overrides_reconfigure:
                    atool.reconfigure(**overrides_reconfigure)
                for attr, value in overrides.items():
                    apply_override(atool, attr, value)
                if "make_patch_sky_plots" in overrides:
                    patch_plots = atool.produce.plot.actions.patch_sky_plots
                    patch_plots.metrics = metrics_plot.copy()
                    for metric_config in patch_plots.metrics.values():
                        metric_config.key = metric_config.key.format(
                            suffix_metric=suffix_metric, band="{band}",
                            name_flux_target="{name_flux_target}",
                            name_flux_ref="{name_flux_ref}",
                        )

                tools_name = {}

                for band_ref, bands_matched in (
                    bands_ref_matched if is_matched else bands_ref_matched_default
                ).items():
                    if band_ref is not None:
                        ref_matched_overrides = {
                            name_attr: value_attr
                            for name_attr in (
                                "key_flux",
                                "name_flux",
                                "name_flux_short",
                                "key_flux_error",
                            )
                            if (value_attr := getattr(ref_matched, name_attr)) is not None and (
                                "{band}" in value_attr
                            )
                        }
                        atool_orig = atool
                        atool = atool_orig.copy()
                        ref_matched_tool = atool.fluxes_default.ref_matched
                        for name_attr, value_attr in ref_matched_overrides.items():
                            setattr(ref_matched_tool, name_attr, value_attr.format(band=band_ref))

                    atool.finalize()
                    produce_plot = atool.produce.plot
                    plots = produce_plot.actions if hasattr(produce_plot, "actions") else {"": produce_plot}
                    for key_plot, plot in plots.items():
                        if hasattr(plot, "publicationStyle"):
                            plot.publicationStyle = True
                        overrides_plot = overrides_produce
                        if key_plot:
                            overrides_plot = overrides_plot.get(key_plot, overrides_plot.get("default", {}))
                        for attr, value in overrides_plot.items():
                            apply_override(plot, attr, value)

                    for column, _ in atool.getInputSchema():
                        if (column == "detect_isPrimary") or (column == "detect_isDeblendedSource") or (
                                column == "refcat_is_pointsource"
                        ):
                            raise ValueError(f"{atool=} has bad {column=} in getInputSchema")
                        if "{band}" in column:
                            for band in bands_ref if column.startswith("refcat_") else bands_lsst:
                                columns.add(column.format(band=band))
                        else:
                            columns.add(column)

                    for band in bands_matched:
                        tools_name[band] = atool

                    if band_ref is not None:
                        for name_attr, value_attr in ref_matched_overrides.items():
                            setattr(ref_matched_tool, name_attr, value_attr)
                        atool = atool_orig

                tools_named[name] = tools_name

            tools_object_type[output_dir] = do_galaxies_only, tools_named

        radecs = {"target": ("coord_ra", "coord_dec")}
        if is_matched:
            radecs["ref"] = (refcat_ra, refcat_dec)
            columns.update(["objectId", "patch"] + tract_keys + patch_keys)
            action_size = LoadVector(vectorKey="sersic_reff_major")
            columns.update({item[0] for item in action_size.getInputSchema()})
            # columns.update({item[0] for item in action_completeness_plot.getInputSchema()})
        for ra, dec in radecs.values():
            columns.update((ra, dec))

        # Some single-tract refcats don't bother with a tract column
        add_tract = "tract" in columns and "tract" not in tract_keys
        if add_tract:
            columns.remove("tract")
        # At some point it must have looped over vis as a band?
        columns_read = tuple(column for column in columns if not column.startswith("vis_"))

        tables = []
        for tract in tracts:
            data = butler.get(
                dataset, skymap=skymap, tract=tract, storageClass="ArrowAstropy",
                parameters={"columns": columns_read}
            )
            if add_tract:
                data["tract"] = tract
            if is_matched:
                no_meas = np.array(np.isfinite(data["objectId"]) != True)
                ra_ref, dec_ref = (data[col][no_meas] for col in radecs["ref"])
                n_no_meas = np.sum(no_meas)
                tract_ref, patch_ref = (np.empty(n_no_meas, dtype=np.uint32) for _ in range(2))
                for idx, (ra, dec) in enumerate(zip(ra_ref, dec_ref)):
                    coord = SpherePoint(ra, dec, degrees)
                    tract_c = skymapInfo.findTract(coord).getId()
                    tract_ref[idx] = tract_c
                    patch_ref[idx] = skymapInfo[tract_c].findPatch(coord).getSequentialIndex()
                if "tract" not in data.colnames:
                    data["tract"] = data["patch"]
                    data["tract"][np.isfinite(data["patch"])] = tract
                tract = np.array(data["tract"])
                tract[no_meas] = tract_ref
                patch = np.array(data["patch"])
                patch[no_meas] = patch_ref
                data["tract"] = tract
                data["patch"] = patch
            tables.append(data)
        data = vstack(tables)

        # Make a naive synthetic LSST VIS band
        for algo in ("cModel", "gaap1p0", "kron", "psf", "sersic"):
            try:
                values = np.sum(
                    [weight*data[f"{band}_{algo}Flux"] for band, weight in weights_lsst_vis.items()],
                    axis=0,
                )/sum_weights_lsst_vis
                data[f"vis_{algo}Flux"] = values
                values = np.sqrt(np.sum(
                    [(weight*data[f"{band}_{algo}FluxErr"])**2 for band, weight in weights_lsst_vis.items()],
                    axis=0,
                ))/sum_weights_lsst_vis
                data[f"vis_{algo}FluxErr"] = values
            except KeyError:
                pass
        if is_matched:
            flux_ref_vis = ref_matched.key_flux.format(band="vis")
            if flux_ref_vis not in data.colnames:
                data[flux_ref_vis] = np.sum(
                    [
                        weight*data[ref_matched.key_flux.format(band=band)]
                        for band, weight in weights_lsst_vis.items()
                    ],
                    axis=0,
                )/sum_weights_lsst_vis
        else:
            data["vis_pixelFlags_saturatedCenter"] = np.sum(
                [data[f"{band}_pixelFlags_saturatedCenter"] for band in weights_lsst_vis],
                axis=0,
            ) > 0

        for output_dir, (do_galaxies_only, tools_named) in tools_object_type.items():
            do_galaxies = (do_galaxies_only is None) or do_galaxies_only

            for name, atool_dict in tools_named.items():
                for band in bands:
                    if (atool := atool_dict.get(band)) is None:
                        atool = atool_dict["default"]
                    overrides_complete = metrics_plot_band.get(band, {})

                    plotInfo["bands"] = [band]
                    schema = atool.getInputSchema()
                    results = atool(data, band=band, plotInfo=plotInfo, skymap=skymapInfo)
                    for name_plot, result in results.items():
                        if isinstance(result, mpl.figure.Figure):
                            suffix = "" if (not "_" in name_plot) else f'_{name_plot.rsplit("_", 1)[0]}'
                            result.savefig(f"{output_dir}/{ref_name}_{skymap}_{field}_{band}_{name}{suffix}.pdf")
