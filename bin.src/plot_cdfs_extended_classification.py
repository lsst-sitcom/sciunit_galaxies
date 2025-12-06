# Make some S/G classification plots for CDFS
# Largely superseded by the notebook, except that it can save plots

import lsst.daf.butler as dafButler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from astropy.table import Table, join
from lsst.obs.base.utils import TableVStack

mpl.rcParams.update({"image.origin": "lower", "font.size": 20})
sns.set_style("darkgrid", {"grid.color": ".6"})

save = True
vertical = False
galaxy_selective = True
ivert = int(vertical)
ihoriz = 1 - ivert
figsize = 10

use_dp1_prep = True
use_cModel = False
use_sn_diff = True
path = "plots_cdfs/"

model_excess_cut = 3 if use_sn_diff else 0.7

collection = "u/dtaranu/DM-50135/DP1_expfit/matched_cdfs" if use_dp1_prep else "u/dtaranu/DM-50135/w_2025_41/matched_cdfs"
butler = dafButler.Butler("/repo/dp1_prep" if use_dp1_prep else "main", collections=collection)

skymap = "lsst_cells_v1"
skymapInfo = butler.get("skyMap", skymap=skymap, collections="skymaps")

matched = butler.get("matched_cdfs_mast_euclid_q1_object", skymap=skymap, tract=5063, collections=collection)

matched_euclid = TableVStack.vstack_handles(
    dafButler.DeferredDatasetHandle(butler, ref, {})
    for ref in butler.query_datasets("matched_euclid_q1_object", skymap=skymap, collections=collection)
)

matched["hst_f_f775+814w"] = np.nanmean([matched["hst_f_f775w"], matched["hst_f_f814w"]], axis=0)


def get_psflike(psfmag_diff, model_excess_cut, delta_ll, delta_ll_cut=0.5):
    is_psflike = psfmag_diff < model_excess_cut
    if (delta_ll is not None) and (delta_ll_cut is not None):
        is_psflike |= (psfmag_diff < 2*model_excess_cut) & (np.abs(delta_ll) < delta_ll_cut)
    return is_psflike


def get_ext_alt(
    log_size, log_sn_psf, psflike, log_size_min=-0.6, log_size_psflike_min=-0.4,
    log_sn_small_thresh=1.5, log_sn_psflike_thresh=1.0,
):
    is_psf = (log_sn_psf > log_sn_small_thresh) & (log_size < log_size_min)
    is_psf |= psflike & (log_sn_psf > log_sn_psflike_thresh) & (log_size < log_size_psflike_min)
    return ~is_psf


def get_psfmag_diff(flux_mod, flux_psf, fluxerr_mod, fluxerr_psf):
    return (flux_mod - flux_psf)/np.sqrt(fluxerr_mod**2 + fluxerr_psf**2)

log_size_min = -0.6
log_size_psflike_min = -0.4
log_sn_small_thresh = 1.5
log_sn_psflike_thresh = 1.0

classes_eye_file = "cdfs_classes.ecsv"
has_classes_eye = os.path.isfile(classes_eye_file)
if has_classes_eye:
    classes = Table.read(classes_eye_file)
    mask_hst = matched["hst_id"].mask
    class_eye = join(
        Table({"hst_id": matched["hst_id"][~mask_hst]}),
        classes,
        join_type="left",
        keep_order=True,
    )["class_eye"]
    values = matched["hst_class_star"].copy()
    values.mask[~mask_hst] = class_eye.mask
    values.value.data[~mask_hst] = class_eye.value.data
    matched["class_eye"] = values
    bad = (matched["hst_flux_radius"] > 3) & (matched["class_eye"] == 1)
    matched["class_eye"][bad == True] = 0.5

cmap = mpl.colormaps["RdYlBu"]
models = ("psf", "sersic") + (("cModel",) if use_cModel else ("exponential",))


def get_arrays(table, models):
    fluxes, fluxerrs = (
        {
            model: np.nansum(
                [table[f"{b}_{model}Flux{suffix}"]**(1.0 + (suffix == "Err")) for b in 'griz'],
                axis=0,
            )**(1.0 - 0.5*(suffix == "Err")) for model in models
        }
        for suffix in ("", "Err")
    )
    mags = {
        model: -2.5*np.log10(np.nanmean([table[f"{b}_{model}Flux"] for b in 'griz'], axis=0)) + 31.4
        for model in models
    }
    flux_psf, flux_ser, flux_alt = (fluxes[x] for x in models)
    fluxerr_psf, fluxerr_ser, fluxerr_alt = (fluxerrs[x] for x in models)
    mag_psf, mag_ser, mag_alt = (mags[x] for x in models)

    return flux_psf, flux_ser, flux_alt, fluxerr_psf, fluxerr_ser, fluxerr_alt, mag_psf, mag_ser, mag_alt

flux_psf, flux_ser, flux_alt, fluxerr_psf, fluxerr_ser, fluxerr_alt, mag_psf, mag_ser, mag_alt = get_arrays(
    matched, models
)

mags_b = {
    model: {b: -2.5*np.log10(matched[f"{b}_{model}Flux"]) + 31.4 for b in 'griz'} for model in models
}

mag_euclid = -2.5*np.log10(matched["euclid_flux_vis_sersic"]) + 31.4

mag_hst = -2.5*np.log10(np.nanmean([matched[f"hst_f_{b}"] for b in ("f435w", "f606w", "f775+814w")], axis=0)) + 31.4
mags_b_hst = {b: -2.5*np.log10(matched[f"hst_f_{b}"]) + 31.4 for b in ("f435w", "f606w", "f775+814w")}

size_ser = np.log10(np.sqrt(0.5*(matched["sersic_reff_x"]**2 + matched["sersic_reff_y"]**2)))
size_alt = np.log10(
    matched["r_bdReD"] if use_cModel else
    (np.sqrt(0.5*(matched["exponential_reff_x"]**2 + matched["exponential_reff_y"]**2)))
)
size_jades = np.log10(matched["jades_A"])

mag_jades = -2.5*np.log10(np.nanmean([matched[f"jades_{b}_SEG"] for b in ("F090W", "F115W")], axis=0)) + 31.4
mags_b_jades = {b: -2.5*np.log10(matched[f"jades_{b}_SEG"]) + 31.4 for b in ("F090W", "F115W")}


def scatter(axis, x, y, xlabel, ylabel, xmin, xmax, ymin, ymax, z, gal, star, cmap, legend: dict = None):
    if legend is None:
        legend = {}
    axis.scatter(x[gal], y[gal], c=cmap(z[gal]), s=10, linewidths=0, label=f"Galaxy {legend.get('galaxy')}")
    axis.scatter(x[star], y[star], c=cmap(z[star]), s=50, linewidths=1, edgecolor="k", label=f"Star {legend.get('star')}")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if legend:
        axis.legend()

mag_width = 1.0

within = (mag_psf > 18) & (mag_psf < 24)

plike_euclid = matched["euclid_point_like_prob"][within]
plike_hst = matched["hst_class_star"][within]
star_thresh = 0.6
gal_thresh = 0.2

star = ((plike_hst > star_thresh) & (plike_euclid > star_thresh))
gal = (plike_hst < gal_thresh)
if galaxy_selective:
    gal &= (plike_euclid < gal_thresh)
else:
    gal |= (plike_euclid < gal_thresh)

z = np.nanmean([plike_euclid, plike_hst], axis=0)

size_lim_min, size_lim_max = -2, 2
sn_min, sn_max = 0.5, 3

fig, ax = plt.subplots(1, 1, figsize=(2*figsize, 2*figsize))
scatter(
    ax,
    np.log10(flux_psf[within]/fluxerr_psf[within]),
    size_ser[within],
    "log10(S/N)", "log10(r_eff/pix)", sn_min, sn_max, size_lim_min, size_lim_max,
    z=z, gal=gal, star=star, cmap=cmap,
)

ax.plot(
    [log_sn_small_thresh, log_sn_small_thresh, sn_max],
    [size_lim_min, log_size_min, log_size_min],
    'k-',
)
ax.plot(
    [log_sn_psflike_thresh, log_sn_psflike_thresh, sn_max],
    [size_lim_min, log_size_psflike_min, log_size_min],
    'k--',
)
ax.set_title("18<mag_psf<24")
fig.tight_layout()
if save:
    fig.savefig(f"{path}sn_size_lsst_cdfs.pdf")


for name_mag, key_mag, mag, flux, fluxerr, size in (
    ("ser", "sersic", mag_ser, flux_ser, fluxerr_ser, size_ser),
    ("cModel" if use_cModel else "exp", "cModel" if use_cModel else "exponential",
     mag_alt, flux_alt, fluxerr_alt, size_alt),
):
    for mag_min in (21, 23):
        mag_max = mag_min + mag_width
        within = (mag_psf > mag_min) & (mag_psf < mag_max)
        plike_euclid = matched["euclid_point_like_prob"][within]
        plike_hst = matched["hst_class_star"][within]
        if has_classes_eye:
            class_eye = matched["class_eye"][within]
        star_thresh = 0.6
        gal_thresh = (0.01 + min(((mag_min < 25)*(25 - mag_min))**1.5/100, 0.29)) if galaxy_selective else (
            star_thresh)
    
        star = ((plike_hst > star_thresh) & (plike_euclid > star_thresh))
        gal = (plike_hst < gal_thresh)
        if galaxy_selective:
            gal &= (plike_euclid < gal_thresh)
        else:
            gal |= (plike_euclid < gal_thresh)
    
        z = np.nanmean([plike_euclid, plike_hst], axis=0)
        if has_classes_eye:
            gal |= (class_eye == 0)
            star |= (class_eye == 1)
            z[class_eye == 1] = 1
            z[class_eye == 0.5] = 0.5
            z[class_eye == 0] = 0

        nrows, ncols = 2 + ivert, 2 + ihoriz
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*figsize, nrows*figsize))
        fig.subplots_adjust(bottom=0.05, left=0.07, top=0.96, right=0.98)
        kwargs = {"z": z, "gal": gal, "star": star, "cmap": cmap}
    
        mpw = mag_psf[within]
        mags_bw = {model: {b: v[within] for b, v in mags_bm.items()} for model, mags_bm in mags_b.items()}
        mpsf_wb = mags_bw["psf"]
        mag_wb = mags_bw[key_mag]
    
        scatter(
            ax[0][0], mpw, size[within], "mag_psf", "log10(reff)", mag_min, mag_max, -2, 2,
            **kwargs
        )
        if use_sn_diff:
            ydiff = (flux[within] - flux_psf[within])/np.sqrt(fluxerr[within]**2 + fluxerr_psf[within]**2)
            ylabel = "(flux - flux_psf)/err"
            ymin, ymax = -5, 10
        else:
            ydiff = matched["exponential_delta_ll_fit_ps"][within]
            ylabel = "LL_exp - LL_psf"
            ymin, ymax = -0.5, 2

        axis = ax[ihoriz][ivert]
        scatter(axis, mpw, ydiff, "mag_psf", ylabel, mag_min, mag_max, ymin, ymax, **kwargs)
        axis.axhline(log_size_min, mag_min, mag_max)
        scatter(
            ax[ivert][ihoriz], mpsf_wb["g"] - mpsf_wb["r"], mpsf_wb["r"] - mpsf_wb["i"],
            "g - r (psf)", "r - i (psf)", -0.5, 2.5, -0.5, 2.5,
            legend={"star": f" (P*>{star_thresh:.2f})", "galaxy": f" (P*<{gal_thresh:.2f})"},
            **kwargs
        )
        scatter(
            ax[1][1], mpsf_wb["i"] - mpsf_wb["z"], mpsf_wb["r"] - mpsf_wb["i"],
            "i - z (psf)", "r - i (psf)", -0.5, 1.1, -0.5, 2.5, **kwargs
        )
        scatter(
            ax[2*ivert][2*ihoriz], mag_wb["g"] - mag_wb["r"], mag_wb["r"] - mag_wb["i"],
            f"g - r ({name_mag})", f"r - i ({name_mag})", -0.5, 2.5, -0.5, 2.5, **kwargs
        )
        scatter(
            ax[2*ivert + ihoriz][2*ihoriz + ivert], mag_wb["i"] - mag_wb["z"], mag_wb["r"] - mag_wb["i"],
            f"i - z ({name_mag})", f"r - i ({name_mag})", -0.5, 1.1, -0.5, 2.5, **kwargs
        )
        fig.suptitle("ECDFS LSST classification")
    
        if save:
            fig.savefig(f"{path}mag_{mag_min}_cdfs_lsst.pdf")
    
        mhst_wb = {b: v[within] for b, v in mags_b_hst.items()}
        mjades_wb = {b: v[within] for b, v in mags_b_jades.items()}
        mag_euclid_w = mag_euclid[within]

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*figsize, nrows*figsize))
        fig.subplots_adjust(bottom=0.05, left=0.07, top=0.96, right=0.98)
        scatter(ax[0][0], mag_hst[within], np.log10(matched["hst_flux_radius"][within]), "HST mag",
                "log10(r_flux)", mag_min, mag_max, 0.25, 1.75, **kwargs)
        scatter(ax[ihoriz][ivert], mag_euclid_w, np.log10(matched["euclid_semimajor_axis"][within]), "Sersic VIS mag",
                "log10(r_eff_maj)", mag_min, mag_max, 0., 1.5, **kwargs)

        scatter(
            ax[ivert][ihoriz], mhst_wb["f435w"] - mhst_wb["f606w"], mhst_wb["f606w"] - mhst_wb["f775+814w"],
            "f435w - f606w", "f606w - f775+814w", -0.1, 2.9, -0.5, 2.5,
            legend={"star": f" (P(*) > {star_thresh:.2f})", "galaxy": f" (P(*) < {gal_thresh:.2f})"}, **kwargs,
        )
        scatter(ax[1][1], mhst_wb["f435w"] - mag_euclid_w, mag_euclid_w - mhst_wb["f775+814w"],
                "f435w - VIS", "VIS - f775+814w", -0.5, 3.5, -0.9, 1.6, **kwargs)

        win_jades = (mag_jades > mag_min) & (mag_jades < mag_max)
        mag_jades_w = mag_jades[within]

        kwargs["z"] = 1.0 - (matched["jades_FLAG_ST"][within] - 1)/32767
    
        scatter(
            ax[2*ivert][2*ihoriz], mag_jades_w, size_jades[within], "JWST mag",
            "log10(r_maj)", mag_min - 0.6, mag_max - 0.6, -1, 0.2, **kwargs
        )
        scatter(
            ax[2*ivert + ihoriz][2*ihoriz + ivert], mag_euclid_w - mjades_wb["F090W"], mjades_wb["F090W"] - mjades_wb["F115W"],
            "VIS - F090W", "F090W - F115W", -0.2, 1.6, -0.2, 1.2, **kwargs,
        )
        fig.suptitle("ECDFS space obs. classification")

        if save:
            fig.savefig(f"{path}mag_{mag_min}_cdfs_{name_mag}.pdf")

if not save:
    plt.show()

star_thresh = 0.4
gal_thresh = 0.7
buffer = 0.02

mags_lo_plot = (21., 22., 23.)
n_mags = len(mags_lo_plot)

for name_mag, mag, flux, fluxerr, size in (
    ("ser", mag_ser, flux_ser, fluxerr_ser, size_ser),
    ("cModel" if use_cModel else "exp", mag_alt, flux_alt, fluxerr_alt, size_alt),
):
    nrows, ncols = n_mags if vertical else 2, 2 if vertical else n_mags
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * figsize, nrows * figsize))
    fig.subplots_adjust(bottom=0.05, left=0.07, top=0.96, right=0.98)

    for idx, mag_min in enumerate(mags_lo_plot):
        mag_max = mag_min + 1
        within = (mag_psf > mag_min) & (mag_psf < mag_max)

        log_sn_psf_w = np.log10(flux_psf[within]/fluxerr_psf[within])
        mag_alt_w = mag_alt[within]
        size_w = size[within]

        if use_sn_diff:
            ydiff = get_psfmag_diff(flux[within], flux_psf[within], fluxerr[within], fluxerr_psf[within])
            psflike = get_psflike(ydiff, model_excess_cut, matched["exponential_delta_ll_fit_ps"][within], delta_ll_cut=1.5)
        else:
            ydiff = matched["exponential_delta_ll_fit_ps"][within]
            psflike = ydiff < model_excess_cut

        ext_alt = get_ext_alt(
            size_w, log_sn_psf_w, psflike,
            log_size_min=log_size_min, log_size_psflike_min=log_size_psflike_min,
            log_sn_small_thresh=log_sn_small_thresh, log_sn_psflike_thresh=log_sn_psflike_thresh,
        )
        ext_ref = matched["refExtendedness"][within]

        plike_euclid = matched["euclid_point_like_prob"][within]
        plike_hst = matched["hst_class_star"][within]

        if has_classes_eye:
            class_eye = matched["class_eye"][within]
            for value_class, value_plot in ((0, -buffer/2), (1, 1 + buffer/2)):
                is_class = class_eye == value_class
                for plike,is_hst in ((plike_euclid, False), (plike_hst, True)):
                    misclassed = is_class & ((plike < 0.5) == (value_class == 1))
                    n_mis = np.sum(misclassed)
                    if n_mis:
                        plike[misclassed] = value_plot
                        print(f"n_misclassed={n_mis} for {value_class=}")

        hst_dx = (np.random.rand(plike_hst.size) - 0.5) * 0.01
        shifted = plike_hst + hst_dx
        bad = (shifted < 0) | (shifted > 0)
        hst_dx[bad] = -hst_dx[bad]

        for idx_typ, (lab, ext) in enumerate((("refExt", ext_ref), (f"{name_mag}Ext", ext_alt))):
            axis = ax[idx if vertical else idx_typ][idx_typ if vertical else idx]

            star = ext == 0
            refstar = (plike_hst + plike_euclid) > (1.0 - star_thresh)
            refgal = (plike_hst + plike_euclid) < (1.0 - gal_thresh)
            star_good = np.sum(star & refstar)/np.sum(refstar)
            gal_good = np.sum(~star & ~refstar)/np.sum(~refstar)

            axis.plot([0, star_thresh], [star_thresh, 0], "-", c="b")
            axis.plot([gal_thresh, 1], [1, gal_thresh], "-", c="r")

            axis.scatter(1.0 - plike_hst[~star & ~refstar] + hst_dx[~star & ~refstar], 1.0 - plike_euclid[~star & ~refstar], s=5, c="r")
            axis.scatter(1.0 - plike_hst[star & refstar] + hst_dx[star & refstar], 1.0 - plike_euclid[star & refstar], s=5, c="b")
            axis.scatter(1.0 - plike_hst[~star & refstar] + hst_dx[~star & refstar], 1.0 - plike_euclid[~star & refstar], s=5, c="r")
            axis.scatter(1.0 - plike_hst[star & ~refstar] + hst_dx[star & ~refstar], 1.0 - plike_euclid[star & ~refstar], s=5, c="b")

            axis.text(star_thresh, 0.0, f"{100*star_good:.1f}%", c="b", va="bottom", ha="left")
            axis.text(gal_thresh, 1.0, f"{100*gal_good:.1f}%", c="r", va="top", ha="right")
            axis.set_xlim(-buffer, 1 + buffer)
            axis.set_ylim(-buffer, 1 + buffer)
            if idx_typ == 0:
                prefix, suffix = ("1 - P* (Euclid)(", ")") if vertical else ("", "")
                (axis.set_ylabel if vertical else axis.set_title)(
                    f"{prefix}{mag_min} < mag < {mag_max}{suffix}"
                )

            if idx == 0:
                prefix, suffix = ("1 - P* (Euclid)(", ")") if not vertical else ("", "")
                (axis.set_title if vertical else axis.set_ylabel)(f"{prefix}{lab}{suffix}")
            if idx == 2:
                if ivert:
                    axis.set_ylabel("1 - class_star (HST)")
            if not ivert:
                axis.set_xlabel("1 - class_star (HST)")
    if save:
        fig.savefig(f"{path}mag_{name_mag}_cdfs_classification.pdf")

if not save:
    plt.show()

sn_min, sn_max = 0.5, 2.5
dsn = 0.05

sig_min, sig_max = (-1, 6) if use_sn_diff else (-1, 5)
dsig = 0.2 if use_sn_diff else 0.1

nx = int(round((sn_max - sn_min)/dsn))
ny = int(round((sig_max - sig_min)/dsig))

nrows, ncols = 1 + 1*ivert, 1 + 1*ihoriz

flux_psf_e, flux_ser_e, _, fluxerr_psf_e, fluxerr_ser_e, _, mag_psf_e, mag_ser_e, _ = get_arrays(
    matched_euclid, models
)
size_ser_e = np.log10(np.sqrt(0.5*(matched_euclid["sersic_reff_x"]**2 + matched_euclid["sersic_reff_y"]**2)))

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize*1.5*ncols, figsize*1.5*nrows))
fig.subplots_adjust(bottom=0.05, left=0.07, top=0.96, right=0.98)

for idx_ax, (name, plike_name, table, flux_psf_i, flux_ser_i, fluxerr_psf_i, fluxerr_ser_i, size_ser_i) in enumerate((
    ("euclid", "euclid_point_like_prob", matched, flux_psf, flux_ser, fluxerr_psf, fluxerr_ser, size_ser),
    ("euclid_full", "refcat_point_like_prob", matched_euclid, flux_psf_e, flux_ser_e, fluxerr_psf_e, fluxerr_ser_e, size_ser_e),
)):
    plike = table[plike_name]
    img = np.zeros((ny, nx))
    sn_lo, sig_lo = sn_min, sig_min

    sn_psf = np.log10(flux_psf_i/fluxerr_psf_i)

    for x in range(nx):
        sn_hi = sn_lo + dsn
        within = (sn_psf >= sn_lo) & (sn_psf < sn_hi)
        mag_psf_w = -2.5*np.log10(flux_psf_i[within]) + 31.4
        size_w = size_ser_i[within]
        within[within] &= (
            size_w < -(log_size_psflike_min - 0.05 * ((mag_psf_w - 23) * (mag_psf_w < 23)))
        ) & ~(
            size_w < log_size_min
        )
        plikew = plike[within]
        if use_sn_diff:
            ydiff = get_psfmag_diff(flux_ser_i[within], flux_psf_i[within], fluxerr_ser_i[within], fluxerr_psf_i[within])
        else:
            ydiff = table["exponential_delta_ll_fit_ps"][within]

        for y in range(ny):
            sig_hi = sig_lo + dsig
            within2 = (ydiff >= sig_lo) & (ydiff < sig_hi)
            if np.sum((values := np.array(plikew[within2])) >= -buffer) > 2:
                img[y, x] = np.sum(values >= 0.5)/np.sum(values >= 0)
            else:
                img[y, x] = np.nan
            sig_lo = sig_hi

        sig_lo = sig_min
        sn_lo = sn_hi

    axis = ax[idx_ax]
    axis.imshow(img, aspect=(sn_max-sn_min)/(sig_max-sig_min), extent=(sn_min, sn_max, sig_min, sig_max),
                vmin=0, vmax=1, cmap="gray")
    axis.set_xlabel("log10(PSF S/N)")
    axis.set_ylabel("Model excess S/N")

fig.tight_layout()

if save:
    fig.savefig(f"{path}euclid_cdfs_classification.pdf")
else:
    plt.show()
