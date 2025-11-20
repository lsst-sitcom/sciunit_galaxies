from lsst.analysis.tools.atools.diffMatched import *
from lsst.analysis.tools.atools.genericBuild import FluxConfig
from lsst.analysis.tools.actions.vector import (
    CalcMomentSize,
    CompositeSelector,
    MatchedTractSelector,
    ReferenceGalaxySelector,
    ReferenceObjectSelector,
    ReferenceStarSelector,
    SetSelector,
)
from lsst.analysis.tools.atools import SizeMagnitudePlot
from lsst.analysis.tools.contexts import CoaddContext
from lsst.analysis.tools.interfaces import NoPlot
import lsst.daf.butler as dafButler
from lsst.geom import degrees, SpherePoint
import lsst.sphgeom as sphgeom
from lsst.utils.plotting import make_figure

from astropy.table import vstack
import matplotlib as mpl
import numpy as np
import skyproj

import os

mpl.rcParams.update({"image.origin": "lower", "font.size": 10, "figure.figsize": (8, 8)})

skymap = "lsst_cells_v1"
tract_patches = {
    4848: (
        20,21,22,23,24,25,26,30,31,32,33,34,35,36,37,40,41,42,43,44,45,46,47,48,
        50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,
        70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
    ),
    4849: (
        24,25,26,27,28,29,33,34,35,36,37,38,39,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,
        60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,
        90,91,92,93,94,95,96,97,98,99
    ),
    5062: (
        0,1,2,3,4,5,10,11,12,13,14,15,20,21,22,23,24,25,30,31,32,33,34,35,36,40,41,42,43,44,45,46,
        50,51,52,53,54,55,60,61,62,63,64,65,70,71,72,73,74,75,80,81,82,83,84,90,91,92,93
    ),
    5063: None,
    5064: (
        4,5,6,7,8,9,14,15,16,17,18,19,24,25,26,27,28,29,35,36,37,38,39,45,46,47,48,49,55,56,57,58,59,
        66,67,68,69,76,77,78,79,87,88,89,97,98,99
    ),
    5280: (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,22),
    5281: (5,6,7,8,9,16,17,18,19,27,28,29),
}
tracts = tuple(tract_patches.keys())

weekly = "w_2025_37"
collection = f"u/dtaranu/DM-50135/{weekly}/matched_cdfs"
butler = dafButler.Butler("/repo/main", collections=collection)
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

weights_lsst_vis = {"g": 0.5, "r": 1.0, "i": 1.0, "z": 1.0}
sum_weights_lsst_vis = sum(weights_lsst_vis.values())
bands_lsst = tuple(weights_lsst_vis.keys()) + ("y",)
bands = ("vis", "y")
bands_color = {"vis": "y"}

reconfigure_color = {"bands_color": bands_color}

selector_all = SetSelector(vectorKeys=["match_candidate"], values=[1])
selector_galaxy = ReferenceGalaxySelector(
    plotLabelValue="Euclid mumax_minus_mag >= -2.6",
    vectorKey="refcat_mumax_minus_mag", threshold=-2.6, op="ge",
)
selector_obj = ReferenceObjectSelector(vectorKey="refcat_spurious_prob", minimum=0, maximum=0.1)
selector_star = ReferenceStarSelector(
plotLabelValue = "Euclid mumax_minus_mag < -2.6",
vectorKey = "refcat_mumax_minus_mag",
threshold = -2.6, op = "le",
)
ref_matched = FluxConfig(
    key_flux="refcat_flux_{band}_sersic",
    name_flux="Reference",
    name_flux_short="Sersic",
    key_flux_error=None,
)
MatchedRefCoaddDiffMagTool.fluxes_default.ref_matched = ref_matched
MatchedRefCoaddCompurityTool.fluxes_default.ref_matched = ref_matched

selector_candidate = CompositeSelector()
selector_candidate.selectors.candidate = SetSelector(
    vectorKeys=("match_candidate", "refcat_match_candidate"),
    values=[1],
)
selector_tract_patches = CompositeSelector(combine_by_and=False)
for tract, patches in tract_patches.items():
    selector_tract = MatchedTractSelector(tract=int(tract), patches=patches) if patches else (
        SetSelector(vectorKeys=["tract", "refcat_tract"], values=[tract]))
    setattr(selector_tract_patches.selectors, f"tract_{tract}", selector_tract)
selector_candidate.selectors.tract_patches = selector_tract_patches

dataset_tools = {
    "object": {
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
    "matched_euclid_q1_object": {
        "completeness": (
            MatchedRefCoaddCompurityTool,
            {
                "mag_bins_plot": {"mag_low_min": mmag_min, "mag_low_max": mmag_max},
                "prep": {
                    "selectors": {"match_candidate": selector_candidate},
                },
                "produce": {
                    "label_shift": -0.15,
                    "legendLocation": "outside upper center",
                    "show_purity": False,
                },
                "selector_all": selector_all,
                "selector_ref_galaxy": selector_galaxy,
                "selector_ref_all": selector_obj,
                "selector_ref_star": selector_star,
            }
        ),
        "ra": (MatchedRefCoaddDiffCoordRaTool, {"produce": kwargs_produce, "coord_ref": "refcat_right_ascension", "coord_ref_cos": "refcat_declination"}),
        "dec": (MatchedRefCoaddDiffCoordDecTool, {"produce": kwargs_produce, "coord_ref": "refcat_declination"}),
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
        "sersic_ra": (MatchedRefCoaddDiffCoordRaTool, {"produce": kwargs_produce, "coord_ref": "refcat_right_ascension", "coord_ref_cos": "refcat_declination"}),
        "sersic_dec": (MatchedRefCoaddDiffCoordDecTool, {"produce": kwargs_produce, "coord_ref": "refcat_declination"}),
    },
}


def apply_override(atool, attr, value):
    if isinstance(value, dict):
        atool_attr = getattr(atool, attr)
        for k, v in value.items():
            apply_override(atool_attr, k, v)
    else:
        setattr(atool, attr, value)


def make_perpatch_completeness_plots(
    atool, data, plotInfo, cmap_patch, cmap_col, band, metric, selector_star,
    action_size: CalcMomentSize | None = None,
    interactive=False,
):
    atool_plot = atool.produce.plot
    atool.produce.plot = NoPlot

    prefix = f"{band}_{metric}_"

    tract_counts = {
        k: (v, {}) for k, v in zip(*np.unique(data["tract"], return_counts=True))
    }

    ra_ref = np.array(data[radecs["ref"][0]])
    dec_ref = np.array(data[radecs["ref"][1]])
    ra_ref_min, ra_ref_max = np.nanmin(ra_ref), np.nanmax(ra_ref)
    if not (ra_ref_max > ra_ref_min):
        ra_ref_min -= 360.
    dec_ref_min, dec_ref_max = np.nanmin(dec_ref), np.nanmax(dec_ref)

    figures = {}
    cmaps_width = 0.045

    if interactive:
        import matplotlib.pyplot as plt

    for name_fig, title, quant, vmin, vmax in (
        ("mag_compl50", f"{band} mag @50% completeness",
         f"{prefix}mag_completeness_50p00_pct", 25.3, 25.7),
        ("compl_23_24", f"completeness 23<{band}<24",
         f"{prefix}completeness_mag23p0", 0.825, 0.925),
    ):
        fig = (plt.figure if interactive else make_figure)(figsize=(8 / (1 - cmaps_width), 8))
        axes = fig.subplots(ncols=3, width_ratios=(1 - 2*cmaps_width, cmaps_width, cmaps_width))
        fig.subplots_adjust(bottom=0.01, left=0.095, right=0.945, top=0.93)
        fig.suptitle(f"{title}; overplot: g-i stars < 17.5 mag, superspreaders (green)")
        fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                cmap=cmap_patch,
            ),
            cax=axes[1],
            extend="both",
        )

        sp = skyproj.GnomonicSkyproj(
            ax=axes[0],
            lon_0=np.nanmedian(ra_ref),
            lat_0=np.nanmedian(dec_ref),
            extent=(ra_ref_min, ra_ref_max, dec_ref_min, dec_ref_max),
        )
        figures[name_fig] = (fig, axes, sp, quant, vmin, vmax)

    for tract, (n_tract, results_border) in tract_counts.items():
        tractInfo = skymapInfo[tract]
        in_tract = data["tract"] == tract
        patch_counts = {
            k: v for k, v in
            zip(*np.unique(data[in_tract]["patch"], return_counts=True))
        }
        in_tracts = {}
        for patch, n_patch in patch_counts.items():
            if n_patch == 0:
                continue

            n_columns, n_rows = tractInfo.getNumPatches()
            patchInfo = tractInfo[patch]
            column, row = patchInfo.getIndex()

            if (results := results_border.get(patch)) is None:
                within = in_tract & (data["patch"] == patch)
                patches_other = []

                left, right = column == 0, column == (n_columns - 1)
                bottom, top = row == 0, row == (n_rows - 1)
                print(patch, row, column, left, right, bottom, top)
                if left or right:
                    bbox = patchInfo._outerBBox
                    pos_y = bbox.getCenterY()
                    pos_x = bbox.getEndX() if right else bbox.getBeginX()
                    if bottom or top:
                        # TODO: figure this out
                        pass
                    else:
                        tractInfo_other = skymapInfo.findTract(tractInfo.wcs.pixelToSky(pos_x, pos_y))
                        tract_other = tractInfo_other.tract_id
                        column_other = 0 if right else n_columns - 1
                        patch_other = tractInfo_other[column_other, row].getSequentialIndex()
                        patches_other.append((tract_other, patch_other))
                        if (in_tract_other := in_tracts.get(tract_other)) is None:
                            in_tract_other = data["tract"] == tract_other
                            in_tracts[tract_other] = in_tract_other
                        within |= (in_tract_other & (data["patch"] == patch_other))
                elif bottom or top:
                    bbox = patchInfo._innerBBox
                    pos_x = bbox.getCenterX()
                    pos_y = bbox.getEndY() if top else bbox.getBeginY()
                    coord_other = tractInfo.wcs.pixelToSky(pos_x, pos_y)
                    tractInfo_other = skymapInfo.findTract(coord_other)
                    tract_other = tractInfo_other.tract_id
                    column_other = skymapInfo[tract_other].findPatch(coord_other).getIndex()[0]
                    if (column_other == 0) or (column_other == (n_columns - 1)):
                        # TODO: figure this out
                        pass
                    else:
                        row_other = 0 if top else n_rows - 1
                        patch_other = tractInfo_other[column_other, row_other].getSequentialIndex()
                        patches_other.append((tract_other, patch_other))
                        if (in_tract_other := in_tracts.get(tract_other)) is None:
                            in_tract_other = data["tract"] == tract_other
                            in_tracts[tract_other] = in_tract_other
                        within |= (in_tract_other & (data["patch"] == patch_other))

            vertices = patchInfo.getInnerSkyPolygon().getVertices()
            clipped = tractInfo.inner_sky_region.clipTo(
                sphgeom.Box(sphgeom.LonLat(vertices[0]), sphgeom.LonLat(vertices[2]))
            )
            lonlats = np.array([
                [x.asDegrees() for x in (lonlat.getA(), lonlat.getB())]
                for lonlat in (clipped.getLon(), clipped.getLat())
            ])
            lons = np.concat((lonlats[0, :], lonlats[0, ::-1]))
            lats = np.repeat(lonlats[1, :], 2)

            if (results is not None) or (n_patch > 500):
                if results is None:
                    subset = data[within]
                    results = atool(subset, band=band, plotInfo=plotInfo, skymap=skymap)
                    for tract_other, patch_other in patches_other:
                        if tract_other in tract_counts:
                            tract_counts[tract_other][1][patch_other] = results

                for fig, axes, sp, quant, vmin, vmax in figures.values():
                    value = results[quant].quantity.value
                    if not np.isfinite(value):
                        continue
                    color = cmap_patch((value - vmin) / (vmax - vmin))
                    sp.draw_polygon(lons, lats, edgecolor="k", facecolor=color, linewidth=0.5)
            else:
                for _, _, sp, *_ in figures.values():
                    sp.draw_polygon(lons, lats, edgecolor="k")

    stars = (data["refExtendedness"] < 0.5) | selector_star(data)
    mags_psf = {band: -2.5 * np.log10(data[f"{band}_psfFlux"]) + 31.4 for band in ("g", "i", "vis")}
    mag_vis = -2.5 * np.log10(data["refcat_flux_vis_sersic"]) + 31.4
    stars_bright = np.ma.where(stars & (np.array(mag_vis < 17.5) | np.array(mags_psf["vis"] < 17.5)))
    gmi = mags_psf["g"][stars_bright] - mags_psf["i"][stars_bright]

    for fig, axes, sp, *_ in figures.values():
        if action_size is not None:
            mag = -2.5*np.log10(np.nanmean([data[f"{b}_sersicFlux"] for b in 'griz'], axis=0)) + 31.4
            size = action_size(data)
            bad = (mag < 21) & (np.log10(size) > (0.7 + 0.1 * np.clip(21 - mag, 0., 10)))
            sp.ax.scatter(
                data["coord_ra"][bad], data["coord_dec"][bad], s=4, c="green", linewidth=0.5, marker="+",
            )
        sp.ax.scatter(
            data["coord_ra"][stars_bright], data["coord_dec"][stars_bright], s=2.5,
            facecolor=cmap_col(1 - gmi/3), edgecolor='k', linewidth=0.1,
        )
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=3, vmax=0, clip=False), cmap=cmap_col),
            cax=axes[2],
            extend="both",
        )
    atool.produce.plot = atool_plot

    return figures

cmap_patch = mpl.colormaps["gray"]
cmap_patch = cmap_patch.from_list(
    "gray",
    cmap_patch(np.linspace(0.16, 0.84, int(round(0.68*256)))),
)
cmap_patch.set_extremes(under=[0.06, 0.06, 0.06, 1], over=[0.94, 0.94, 0.94, 1])
cmap_col = mpl.colormaps["RdYlBu"]
cmap_col.set_extremes(bad=[0.5, 0.5, 0.5, 1], under=[0.4, 0.0, 0.1, 1.0], over=[0.1, 0.15, 0.45, 1.0])

for dataset, tools in dataset_tools.items():
    tools_object_type = {}
    columns = {f"{band}_psfFlux" for band in bands_lsst}
    # stars otherwise
    for do_galaxies_only in (True, False, None):
        do_galaxies = do_galaxies_only == True
        suffix_folder = "all" if (do_galaxies_only is None) else ("galaxies" if do_galaxies else "stars")
        output_dir = f"dp2_cdfs_plots_{weekly}_{suffix_folder}"
        tools_named = {}

        if not os.path.exists(output_dir):
            print(f"{output_dir=} does not exist; making it now")
            os.mkdir(output_dir)
        elif not os.path.isdir(output_dir):
            raise RuntimeError(f"{output_dir=} exists but is not a directory")

        reconfigure_diff_matched_defaults(
            config=None,
            context="custom",
            key_flux_meas="cmodel_err",
            use_any=do_galaxies_only is None,
            use_galaxies=do_galaxies,
            use_stars=not do_galaxies_only,
        )

        plotInfo["tableName"] = dataset
        # data = butler.get(dataset, skymap=skymap, tract=tract, storageClass="ArrowAstropy")
        for name, (class_tool, overrides_orig) in tools.items():
            overrides = overrides_orig.copy()
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
            atool.finalize()
            produce_plot = atool.produce.plot
            plots = produce_plot.actions if hasattr(produce_plot, "actions") else [produce_plot]
            for plot in plots:
                plot.publicationStyle = True
                for attr, value in overrides_produce.items():
                    apply_override(plot, attr, value)
            tools_named[name] = atool
            for column, _ in atool.getInputSchema():
                if "{band}" in column:
                    for band in bands if column.startswith("refcat") else bands_lsst:
                        columns.add(column.format(band=band))
                else:
                    columns.add(column)
        tools_object_type[output_dir] = do_galaxies_only, tools_named

    radecs = {"target": ("coord_ra", "coord_dec")}
    is_matched = dataset.startswith("matched_")
    if is_matched:
        radecs["ref"] = ("refcat_right_ascension", "refcat_declination")
        columns.update(("objectId", "tract", "patch", "refcat_tract", "refcat_patch"))
        action_size = CalcMomentSize(colXx="sersic_reff_x", colYy="sersic_reff_y", colXy="sersic_rho",
                                     is_covariance=False, sizeType="determinant")
        columns.update({item[0] for item in action_size.getInputSchema()})
    for ra, dec in radecs.values():
        columns.update((ra, dec))

    columns_read = tuple(column for column in columns if not column.startswith("vis_"))

    tables = []
    for tract in tracts:
        data = butler.get(
            dataset, skymap=skymap, tract=tract, storageClass="ArrowAstropy",
            parameters={"columns": columns_read}
        )
        if dataset.startswith("matched_euclid_q1_object"):
            no_meas = data["objectId"].mask == True
            ra_ref, dec_ref = (data[col][no_meas] for col in radecs["ref"])
            tract_ref, patch_ref = (np.empty(np.sum(no_meas), dtype=np.uint32) for _ in range(2))
            for idx, (ra, dec) in enumerate(zip(ra_ref, dec_ref)):
                coord = SpherePoint(ra, dec, degrees)
                tract_c = skymapInfo.findTract(coord).getId()
                tract_ref[idx] = tract_c
                patch_ref[idx] = skymapInfo[tract_c].findPatch(coord).getSequentialIndex()
            tract = np.array(data["tract"])
            tract[no_meas] = tract_ref
            patch = np.array(data["patch"])
            patch[no_meas] = patch_ref
            data["tract"] = tract
            data["patch"] = patch
        tables.append(data)
    data = vstack(tables)
    if dataset.startswith("matched_euclid_q1_object"):
        data = data[selector_candidate(data)]

    # Make a naive synthetic LSST VIS band
    for algo in ("cModel", "gaap1p0", "gaap3p0", "kron", "psf", "sersic"):
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

    for output_dir, (do_galaxies_only, tools_named) in tools_object_type.items():
        do_galaxies = (do_galaxies_only is None) or do_galaxies_only
        suffix_metric = "all" if (do_galaxies_only is None) else ("resolved" if do_galaxies else "unresolved")

        for name, atool in tools_named.items():
            for band in bands:
                plotInfo["bands"] = [band]
                results = atool(data, band=band, plotInfo=plotInfo, skymap=skymap)
                for name_plot, result in results.items():
                    if isinstance(result, mpl.figure.Figure):
                        suffix = "" if (not "_" in name_plot) else f'_{name_plot.rsplit("_", 1)[0]}'
                        result.savefig(f"{output_dir}/euclid_{skymap}_cdfs_{band}_{name}{suffix}.pdf")
                if name == "completeness":
                    figures = make_perpatch_completeness_plots(
                        atool, data, plotInfo, cmap_patch, cmap_col, band,
                        metric=f"detect_cModel_vs_Sersic_{suffix_metric}",
                        selector_star=selector_star,
                    )
                    for name_fig, (figure, *_) in figures.items():
                        figure.savefig(f"{output_dir}/euclid_{skymap}_cdfs_{band}_{name_fig}.pdf")
