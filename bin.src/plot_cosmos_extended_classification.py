# Make some S/G classification plots for CDFS
# Largely superseded by the notebook, except that it can save plots

import astropy.units as u
import lsst.daf.butler as dafButler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.rcParams.update({"image.origin": "lower", "font.size": 20})
sns.set_style("darkgrid", {"grid.color": ".6"})

save = True
vertical = False
galaxy_selective = True
ivert = int(vertical)
ihoriz = 1 - ivert
figsize = 5

use_cModel = False
use_sn_diff = True

model_excess_cut = 3 if use_sn_diff else 0.7

collection = "u/dtaranu/DM-50135/w_2025_43/matched_cosmos"
butler = dafButler.Butler("main", collections=collection)
dataset = "matched_cosmos_mast_object"
path = "plots_cosmos/"

skymap = "lsst_cells_v1"
skymapInfo = butler.get("skyMap", skymap=skymap, collections="skymaps")
tract = 9813

matched = butler.get(dataset, skymap=skymap, tract=tract, collections=collection)

col_plike_hst = "hst_class_star"
col_plike_jwst = "cosmos2025_type"
delta_ll_cut = None


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

log_size_min = -0.6
log_size_psflike_min = -0.4
log_sn_small_thresh = 1.5
log_sn_psflike_thresh = 1.0

cmap = mpl.colormaps["RdYlBu"]
models = ("psf", "sersic") + (("cModel",) if use_cModel else ("exponential",))
size_jwst = np.log10(matched["cosmos2025_radius_sersic"].to(u.arcsec)/u.arcsec)
size_hst = np.log10(matched["hst_flux_radius"])


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

psfl_size_intercept = np.nanmedian(mag_psf[((flux_psf/fluxerr_psf) > 9) & (flux_psf/fluxerr_psf < 12)])

mags_b = {
    model: {b: -2.5*np.log10(matched[f"{b}_{model}Flux"]) + 31.4 for b in 'griz'} for model in models
}

mag_hst = -2.5*np.log10(matched["hst_flux_auto"]) + 31.4

mag_jwst = -2.5*np.log10(matched["cosmos2025_flux_model_hst-f814w"]) + 31.4
mags_b_jwst = {
    b: -2.5*np.log10(matched[f"cosmos2025_flux_model_{b}"]) + 31.4
    for b in ("hst-f814w", "f115w", "f150w", "f277w")
}

size_ser = np.log10(np.sqrt(0.5*(matched["sersic_reff_x"]**2 + matched["sersic_reff_y"]**2)))
size_alt = np.log10(
    matched["r_bdReD"] if use_cModel else
    (np.sqrt(0.5*(matched["exponential_reff_x"]**2 + matched["exponential_reff_y"]**2)))
)


def scatter(
    axis, x, y, xlabel, ylabel, xmin, xmax, ymin, ymax, z, gal, star, cmap, legend: dict = None,
    gal_first=True, save=False,
):
    if legend is None:
        legend = {}
    kwargs_gal = {
        "x": x[gal], "y": y[gal], "c": cmap(z[gal]), "s": 6 if save else 10,
        "linewidths": 0, "label": f"Galaxy {legend.get('galaxy')}",
    }
    if gal_first:
        axis.scatter(**kwargs_gal)
    axis.scatter(
        x[star], y[star], c=cmap(z[star]), s=30 if save else 50, linewidths=1, edgecolor="k",
        label=f"Star {legend.get('star')}",
    )
    if not gal_first:
        axis.scatter(**kwargs_gal)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if legend:
        axis.legend()

mag_width = 1.0

within = (mag_psf > 18) & (mag_psf < 26)
plike_hst = matched[col_plike_hst][within]
plike_jwst = matched[col_plike_jwst][within]
star_thresh = 0.6
gal_thresh = 0.2

star = ((plike_jwst > star_thresh) & (plike_hst > star_thresh) & (plike_jwst <= 1) & (plike_hst <= 1))
gal = (plike_jwst < gal_thresh) & (plike_jwst >= 0)
if galaxy_selective:
    gal &= (plike_hst < gal_thresh) & (plike_hst >= 0)
else:
    gal |= (plike_hst < gal_thresh) & (plike_hst >= 0)

z = np.nanmean([plike_hst, plike_jwst], axis=0)

size_lim_min, size_lim_max = -2, 2
sn_min, sn_max = 0.5, 3

fig, ax = plt.subplots(1, 1, figsize=(2.5*figsize, 2.5*figsize))
scatter(
    ax,
    np.log10(flux_psf[within]/fluxerr_psf[within]),
    size_ser[within],
    "log10(S/N)", "log10(r_eff/pix)", sn_min, sn_max, size_lim_min, size_lim_max,
    z=z, gal=gal, star=star, cmap=cmap,
    gal_first=False,
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
ax.set_title("18<mag_psf<26")
fig.tight_layout()
if save:
    fig.savefig(f"{path}sn_size_lsst_cosmos.pdf")


for name_mag, key_mag, mag, flux, fluxerr, size in (
    ("ser", "sersic", mag_ser, flux_ser, fluxerr_ser, size_ser),
    ("cModel" if use_cModel else "exp", "cModel" if use_cModel else "exponential",
     mag_alt, flux_alt, fluxerr_alt, size_alt),
):
    for mag_min in (21, 23):
        mag_max = mag_min + mag_width
        within = (mag_psf > mag_min) & (mag_psf < mag_max)
        plike_hst = matched[col_plike_hst][within]
        plike_jwst = matched[col_plike_jwst][within]

        star_thresh = 0.6
        gal_thresh = (0.01 + min(((mag_min < 25)*(25 - mag_min))**1.5/100, 0.29)) if galaxy_selective else (
            star_thresh)

        star = ((plike_jwst > star_thresh) & (plike_hst > star_thresh) & (plike_jwst <= 1) & (plike_hst <= 1))
        gal = (plike_jwst < gal_thresh) & (plike_jwst >= 0)
        if galaxy_selective:
            gal &= (plike_hst < gal_thresh) & (plike_hst >= 0)
        else:
            gal |= (plike_hst < gal_thresh) & (plike_hst >= 0)

        z = np.nanmean([plike_hst, plike_jwst], axis=0)

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
            ydiff = matched["exponential_delta_lnL_fit_ps"][within]
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
        fig.suptitle("COSMOS LSST classification")

        if save:
            fig.savefig(f"{path}mag_{mag_min}_cosmos_lsst.pdf")

        mjwst_wb = {b: v[within] for b, v in mags_b_jwst.items()}
        mag_hst_w = mag_hst[within]
        mag_jwst_w = mag_jwst[within]

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(ncols*figsize, nrows*figsize))
        fig.subplots_adjust(bottom=0.05, left=0.07, top=0.96, right=0.98)
        scatter(ax[0][0], mag_hst_w, size_hst[within], "HST auto mag",
                "log10(r_flux)", mag_min, mag_max, 0.25, 1.75, **kwargs)
        scatter(ax[ihoriz][ivert], mag_jwst_w, size_jwst[within], "HST Sersic mag",
                "log10(r_eff)", mag_min, mag_max, -3, 1.0, **kwargs)

        scatter(
            ax[ivert][ihoriz], mjwst_wb["hst-f814w"] - mjwst_wb["f115w"], mjwst_wb["f115w"] - mjwst_wb["f150w"],
            "hst-f814w - f115w", "f115w - f150w", -0.1, 2.9, -0.5, 2.5,
            legend={"star": f" (P(*) > {star_thresh:.2f})", "galaxy": f" (P(*) < {gal_thresh:.2f})"}, **kwargs,
        )
        scatter(
            ax[1][1], mag_hst_w - mjwst_wb["f115w"], mjwst_wb["f115w"] - mjwst_wb["f150w"],
            "f814w - f115w", "f115w - f150w", -0.1, 2.9, -0.5, 2.5, **kwargs,
        )
        fig.suptitle("COSMOS space obs. classification")

        if save:
            fig.savefig(f"{path}mag_{mag_min}_cosmos_{name_mag}.pdf")

if not save:
    plt.show()

star_thresh = 0.4
gal_thresh = 0.7
buffer = 0.02

def get_psfmag_diff(flux_mod, flux_psf, fluxerr_mod, fluxerr_psf):
    return (flux_mod - flux_psf)/np.sqrt(fluxerr_mod**2 + fluxerr_psf**2)


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
            kwargs = {
                "delta_ll": matched["exponential_delta_lnL_fit_ps"][within] if delta_ll_cut is not None else None,
                "delta_ll_cut": delta_ll_cut,
            }
            psflike = get_psflike(ydiff, model_excess_cut, **kwargs)
        else:
            ydiff = matched["exponential_delta_lnL_fit_ps"][within]
            psflike = ydiff < model_excess_cut

        ext_alt = get_ext_alt(
            size_w, log_sn_psf_w, psflike, log_size_min=log_size_min, log_size_psflike_min=log_size_psflike_min,
        )
        ext_ref = matched["refExtendedness"][within]

        plike_hst = matched[col_plike_hst][within]
        plike_jwst = matched[col_plike_jwst][within]

        hst_dx = (np.random.rand(plike_jwst.size) - 0.5) * 0.01
        shifted = plike_jwst + hst_dx
        bad = (shifted < 0) | (shifted > 0)
        hst_dx[bad] = -hst_dx[bad]

        for idx_typ, (lab, ext) in enumerate((("refExt", ext_ref), (f"{name_mag}Ext", ext_alt))):
            axis = ax[idx if vertical else idx_typ][idx_typ if vertical else idx]

            star = ext == 0
            refstar = (plike_jwst + plike_hst) > (1.0 - star_thresh)
            refgal = (plike_jwst + plike_hst) < (1.0 - gal_thresh)
            star_good = np.sum(star & refstar)/np.sum(refstar)
            gal_good = np.sum(~star & ~refstar)/np.sum(~refstar)

            axis.plot([0, star_thresh], [star_thresh, 0], "-", c="b")
            axis.plot([gal_thresh, 1], [1, gal_thresh], "-", c="r")

            axis.scatter(1.0 - plike_jwst[~star & ~refstar] + hst_dx[~star & ~refstar], 1.0 - plike_hst[~star & ~refstar], s=5, c="r")
            axis.scatter(1.0 - plike_jwst[star & refstar] + hst_dx[star & refstar], 1.0 - plike_hst[star & refstar], s=5, c="b")
            axis.scatter(1.0 - plike_jwst[~star & refstar] + hst_dx[~star & refstar], 1.0 - plike_hst[~star & refstar], s=5, c="r")
            axis.scatter(1.0 - plike_jwst[star & ~refstar] + hst_dx[star & ~refstar], 1.0 - plike_hst[star & ~refstar], s=5, c="b")

            axis.text(star_thresh, 0.0, f"{100*star_good:.1f}%", c="b", va="bottom", ha="left")
            axis.text(gal_thresh, 1.0, f"{100*gal_good:.1f}%", c="r", va="top", ha="right")
            axis.set_xlim(-buffer, 1 + buffer)
            axis.set_ylim(-buffer, 1 + buffer)
            if idx_typ == 0:
                prefix, suffix = ("1 - P* (HST)(", ")") if vertical else ("", "")
                (axis.set_ylabel if vertical else axis.set_title)(
                    f"{prefix}{mag_min} < mag < {mag_max}{suffix}"
                )

            if idx == 0:
                prefix, suffix = ("1 - P* (HST)(", ")") if not vertical else ("", "")
                (axis.set_title if vertical else axis.set_ylabel)(f"{prefix}{lab}{suffix}")
            if idx == 2:
                if ivert:
                    axis.set_ylabel("1 - class_star (JWST)")
            if not ivert:
                axis.set_xlabel("1 - class_star (JWST)")
    if save:
        fig.savefig(f"{path}mag_{name_mag}_cosmos_classification.pdf")

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
    matched, models
)
size_ser_e = np.log10(np.sqrt(0.5*(matched["sersic_reff_x"]**2 + matched["sersic_reff_y"]**2)))

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize*1.5*ncols, figsize*1.5*nrows))
fig.subplots_adjust(bottom=0.05, left=0.07, top=0.96, right=0.98)

for idx_ax, (name, plike_name, table, flux_psf_i, flux_ser_i, fluxerr_psf_i, fluxerr_ser_i, size_ser_i) in enumerate((
    ("jwst", col_plike_jwst, matched, flux_psf, flux_ser, fluxerr_psf, fluxerr_ser, size_jwst),
    ("hst", col_plike_hst, matched, flux_psf, flux_ser, fluxerr_psf, fluxerr_ser, size_ser),
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
            ydiff = table["exponential_delta_lnL_fit_ps"][within]

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
    fig.savefig(f"{path}hst_cdfs_classification.pdf")
else:
    plt.show()
