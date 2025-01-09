# Plot model photometry of stars in CDFS compared to DES catalogs
# This is to investigate discrepancies in DECaLS r & i-band photometry

import lsst.daf.butler as dafButler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({"image.origin": "lower", 'font.size': 18})

butler = dafButler.Butler("/repo/embargo")
skymap = "lsst_cells_v1"
tract = 5063
bands = ("g", "r", "i")
save = True

for catalog, label in (("decals_dr10", "DECaLS DR10"), ("des_y3gold", "DES Y3Gold")):
    is_decals = catalog.startswith("decals")
    matched = butler.get(
        f"matched_{catalog}_objectTable_tract", tract=tract, skymap=skymap,
        collections="u/dtaranu/DM-47234/20241101_20241120/match",
    )

    mag_max_r = 20
    mags_meas = {}
    mags_ref = {}
    column_flux_ref = "refcat_flux_{band}" if is_decals else "refcat_sof_cm_flux_{band}"

    for band in bands:
        mags_meas[band] = -2.5*np.log10(matched[f"{band}_cModelFlux"]) + 31.4
        mags_ref[band] = -2.5*np.log10(matched[column_flux_ref.format(band=band)]) + 31.4

    is_bright_star = (matched["refExtendedness"] < 0.5) & (mags_meas["r"] < mag_max_r)
    is_psf_model = (
        (matched["refcat_type"] == "PSF") if is_decals
        else matched["refcat_extended_class_mash_sof"]
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(24, 16))
    plt.suptitle(f"CModel vs {label} stellar photometry {tract=}, r < {mag_max_r}")

    for idx, band in enumerate(bands):
        mag_meas = mags_meas[band]
        mag_ref = mags_ref[band]
        axis = ax[0][idx]
        axis.scatter(mag_meas[is_bright_star], 1000*(mag_ref[is_bright_star] - mag_meas[is_bright_star]))
        axis.set_xlabel(f"{band} CModel mag")
        axis.set_ylabel(f"{label} - CModel (mmag)")
        axis.set_xlim((15, 21))
        axis.set_ylim(-200, 200)

    (gmr_meas, rmi_meas), (gmr_ref, rmi_ref) = (
        tuple(
            mags[band1][is_bright_star] - mags[band2][is_bright_star]
            for band1, band2 in (("g", "r"), ("r", "i"))
        )
        for mags in (mags_meas, mags_ref)
    )
    is_psf_model_bright = is_psf_model[is_bright_star]

    for axis in (ax[1][0], ax[1][2]):
        axis.scatter(gmr_meas, rmi_meas, c="gray", label="ComCam CModel")

    for axis in ax[1][1:3]:
        if is_decals:
            for color, label_line, is_psf in (
                ("midnightBlue", f"{label} PSF model", True),
                ("firebrick", "Non-PSF model", False),
            ):
                kwargs_ax = {"c": color, "label": label_line}
                select = is_psf_model_bright == is_psf
                axis.scatter(gmr_ref[select], rmi_ref[select], **kwargs_ax)
        else:
            for color, label_line, val_lo, val_hi in (
                ("midnightBlue", f"{label} likely PSF", -0.5, 0.5),
                ("darkviolet", f"{label} maybe PSF", 0.5, 1.5),
                ("firebrick", f"{label} prob. not PSF", 1.5, 3.5),
            ):
                kwargs_ax = {"c": color, "label": label_line}
                select = (is_psf_model_bright >= val_lo) & (is_psf_model_bright < val_hi)
                axis.scatter(gmr_ref[select], rmi_ref[select], **kwargs_ax)

    for axis in ax[1]:
        axis.legend()
    ax[1][2].set_xlim((-0.7, 2))
    ax[1][2].set_ylim((-0.7, 2.5))

    for axis in ax[1]:
        axis.set_xlabel("g-r")
        axis.set_ylabel("r-i")

    fig.tight_layout()
    if save:
        fig.savefig(f"stellar_phot_{skymap}_{tract}_{catalog}.png")
    else:
        plt.show()
