from lsst.analysis.tools.atools.diffMatched import *
from lsst.analysis.tools.actions.vector.selectors import InjectedGalaxySelector, InjectedStarSelector
from lsst.analysis.tools.atools import SizeMagnitudePlot
from lsst.analysis.tools.contexts import CoaddContext
from lsst.analysis.tools.interfaces import NoPlot
import lsst.daf.butler as dafButler
from lsst.geom import degrees, SpherePoint
import lsst.sphgeom as sphgeom
from lsst.utils.plotting import make_figure

from astropy.table import vstack
import healsparse as hsp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skyproj

import os

skymap = "lsst_cells_v1"

bo_hsp = hsp.HealSparseMap.read('/sdf/home/b/bclevine/A360/masks_Rubin_SV_38_7.hs')
hand_hsp = hsp.HealSparseMap.read('/sdf/home/b/bclevine/A360/A360_maskmap_hsp_1024_16384.fits')
sfd_hsp = hsp.HealSparseMap.read('/sdf/home/b/bclevine/A360/A360_sfd_map_hsp_1024_16384.fits')

InjectedGalaxySelector.key_class.default = "ref_r_comp1_source_type"
InjectedStarSelector.key_class.default = "ref_r_comp1_source_type"

collection = "u/dtaranu/DM-50124/injected_dp1_prep/matched"
butler = dafButler.Butler("/repo/dp1_prep", collections=collection)
skymapInfo = butler.get("skyMap", skymap=skymap)

tracts = (10463, 10464, 10704)

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

bands = ("z", "i", "r", "g")
bands_color = {'g': 'r,i', 'r': 'i', 'i': 'z'}

reconfigure_color = {"bands_color": bands_color}

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
                    "show_purity": False,
                },
                "reconfigure": {"pany": False},
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
        "color_cmodel": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "reconfigure": reconfigure_color}),
        "color_gaap": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "gaap1p0_err", "reconfigure": reconfigure_color}),
        "color_psf": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "psf_err", "reconfigure": reconfigure_color}),
        "color_sersic": (MatchedRefCoaddDiffColorTool, {"produce": kwargs_produce, "mag_y1": "sersic_err", "reconfigure": reconfigure_color}),
        "color_chi_cmodel": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "reconfigure": reconfigure_color}),
        "color_chi_gaap": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "gaap1p0_err", "reconfigure": reconfigure_color}),
        "color_chi_psf": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "psf_err", "reconfigure": reconfigure_color}),
        "color_chi_sersic": (MatchedRefCoaddChiColorTool, {"produce": kwargs_produce_chi, "mag_y1": "sersic_err", "reconfigure": reconfigure_color}),
        "sersic_ra": (MatchedRefCoaddDiffCoordRaTool, {"produce": kwargs_produce}),
        "sersic_dec": (MatchedRefCoaddDiffCoordDecTool, {"produce": kwargs_produce}),
    },
}


def apply_override(atool, attr, value):
    if isinstance(value, dict):
        atool_attr = getattr(atool, attr)
        for k, v in value.items():
            apply_override(atool_attr, k, v)
    else:
        setattr(atool, attr, value)

for dataset, tools in dataset_tools.items():
    tools_object_type = [{}, {}]
    columns = set()
    # stars otherwise
    for do_galaxies in (True, False):
        suffix_folder = "_galaxies" if do_galaxies else "_stars"
        output_dir = f"a360_plots{suffix_folder}"
        if not os.path.exists(output_dir):
            print(f"{output_dir=} does not exist; making it now")
            os.mkdir(output_dir)
        elif not os.path.isdir(output_dir):
            raise RuntimeError(f"{output_dir=} exists but is not a directory")

        reconfigure_diff_matched_defaults(
            config=None,
            context="injection",
            key_flux_meas="cmodel_err",
            use_any=False,
            use_galaxies=do_galaxies,
            use_stars=not do_galaxies,
        )

        plotInfo["tableName"] = dataset
        # data = butler.get(dataset, skymap=skymap, tract=tract, storageClass="ArrowAstropy")
        for name, (class_tool, overrides_orig) in tools.items():
            overrides = overrides_orig.copy()
            atool = class_tool()
            overrides_produce = overrides.pop("produce", {})
            overrides_reconfigure = overrides.pop("reconfigure", {})
            if overrides_reconfigure:
                if name == "completeness":
                    overrides_reconfigure["use_galaxies"] = do_galaxies
                    overrides_reconfigure["use_stars"] = not do_galaxies
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
            tools_object_type[do_galaxies][name] = atool
            for column, _ in atool.getInputSchema():
                if "{band}" in column:
                    for band in bands:
                        columns.add(column.format(band=band))
                else:
                    columns.add(column)
                    if column.startswith("u_"):
                        print(name, column)

    if dataset == "object_all":
        columns.add("detect_isTractInner")
    radecs = {"target": ("coord_ra", "coord_dec")}
    is_matched = dataset.startswith("matched_")
    if is_matched:
        radecs["ref"] = ("ref_ra", "ref_dec")
        columns.update(("objectId", "tract", "patch", "ref_tract"))
    for ra, dec in radecs.values():
        columns.update((ra, dec))

    tables = []
    for tract in tracts:
        data = butler.get(
            dataset, skymap=skymap, tract=tract, storageClass="ArrowAstropy", parameters={"columns": columns}
        )
        if dataset == "object_all":
            data = data[data["detect_isTractInner"] == True]
        elif dataset.startswith("matched_injected_deep_coadd_"):
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

    mask = np.zeros(len(data), dtype=bool)
    masks_cat_type = {}

    for cat_type, (ra, dec) in radecs.items():
        star_mask, hand_mask, sfd_mask = (np.zeros(len(data), dtype=bool) for _ in range(3))
        good = np.isfinite(data[ra]) & np.isfinite(data[dec])
        ra_good, dec_good = (data[col][good].value for col in (ra, dec))
        star_mask[good] = ~bo_hsp.get_values_pos(ra_good, dec_good, lonlat=True)
        hand_mask[good] = ~hand_hsp.get_values_pos(ra_good, dec_good, lonlat=True)
        sfd_mask[good] = sfd_hsp.get_values_pos(ra_good, dec_good, lonlat=True) <= 0.15
        mask_typed = (star_mask & hand_mask & sfd_mask)
        mask |= mask_typed
        if is_matched and (cat_type == "target"):
            masks_cat_type[cat_type] = mask_typed

    data_masked = data[mask]

    for do_galaxies in (False, True):
        for name, atool in tools_object_type[do_galaxies].items():
            for band in bands:
                plotInfo["bands"] = [band]
                results = atool(data_masked, band=band, plotInfo=plotInfo, skymap=skymap)
                for name_plot, result in results.items():
                    if isinstance(result, mpl.figure.Figure):
                        suffix = "" if (not "_" in name_plot) else f'_{name_plot.rsplit("_", 1)[0]}'
                        result.savefig(f"{output_dir}/injected_{skymap}_a360_{band}_{name}{suffix}.pdf")
                if name == "completeness":
                    atool_plot = atool.produce.plot
                    atool.produce.plot = NoPlot

                    prefix = f"{band}_detect_cModel_vs_true_{'resolved' if do_galaxies else 'unresolved'}_"
                    cmap = mpl.colormaps["viridis"]
                    cmapv = getattr(cmap, "colors", cmap(range(256)))
                    cmap_min = 16
                    cmap_range = 255 - 2*cmap_min

                    tract_counts = {
                        k: v for k, v in zip(*np.unique(data_masked["tract"], return_counts=True))
                    }

                    ra_ref = np.array(data_masked[radecs["ref"][0]])
                    dec_ref = np.array(data_masked[radecs["ref"][1]])
                    ra_ref_min, ra_ref_max = np.nanmin(ra_ref), np.nanmax(ra_ref)
                    if not (ra_ref_max > ra_ref_min):
                        ra_ref_min -= 360.
                    dec_ref_min, dec_ref_max = np.nanmin(dec_ref), np.nanmax(dec_ref)

                    # fig = make_figure()
                    figures = {}
                    cmap_width = 0.05

                    for name_fig, title, quant, vmin, vmax in (
                        ("mag_compl50", f"{band} mag @50% completeness",
                         f"{prefix}mag_completeness_50p00_pct", 25.2, 25.7),
                        ("compl_24_25", f"completeness 24<{band}<25",
                         f"{prefix}completeness_mag24p0", 0.78, 0.93),
                    ):
                        fig = plt.figure(figsize=(10/(1 - cmap_width), 10))
                        axes = fig.subplots(ncols=2, width_ratios=(1 - cmap_width, cmap_width))
                        fig.suptitle(title)
                        fig.colorbar(
                            mpl.cm.ScalarMappable(
                                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                                cmap=cmap,
                            ),
                            cax=axes[1],
                        )
                        sp = skyproj.GnomonicSkyproj(
                            ax=axes[0],
                            lon_0=np.nanmedian(ra_ref),
                            lat_0=np.nanmedian(dec_ref),
                            extent=(ra_ref_min, ra_ref_max, dec_ref_min, dec_ref_max),
                        )
                        figures[name_fig] = (axes, sp, quant, vmin, vmax)

                    for tract, n_tract in tract_counts.items():
                        tractInfo = skymapInfo[tract]
                        in_tract = data_masked["tract"] == tract
                        patch_counts = {
                            k: v for k, v in
                            zip(*np.unique(data_masked[in_tract]["patch"], return_counts=True))
                        }
                        for patch, n_patch in patch_counts.items():
                            if n_patch > 10:
                                subset = data_masked[in_tract & (data_masked["patch"] == patch)]
                                patchInfo = tractInfo[patch]

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

                                if n_patch > 500:
                                    results = atool(subset, band=band, plotInfo=plotInfo, skymap=skymap)

                                    for axes, sp, quant, vmin, vmax in figures.values():
                                        value = results[quant].quantity.value
                                        if value < vmin:
                                            color = cmapv[0]
                                        elif value > vmax:
                                            color = cmapv[255]
                                        else:
                                            color = cmapv[int(round(
                                                cmap_min + cmap_range*(value - vmin)/(vmax - vmin)
                                            ))]
                                        sp.draw_polygon(lons, lats, edgecolor="k", facecolor=color, linewidth=0.5)
                                else:
                                    print(tract, patch, n_patch)
                                    for _, sp, *_ in figures.values():
                                        sp.draw_polygon(lons, lats, edgecolor="k")
                            else:
                                print(tract, patch, n_patch)
