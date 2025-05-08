import astropy.units as u
import lsst.gauss2d as g2d
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .lsst import scale_lsst_deg


def _scatter_multi(axis, x, y, kwargs_scatter_list):
    for kwargs_scatter in kwargs_scatter_list:
        axis.scatter(x, y, **kwargs_scatter)


def plot_external_matches(
    img_rgb_lsst, img_rgb_ext, extent, matched_in,
    id_ext: str,
    bands_fluxes_ext: dict[str, str],
    figsize_ax: float = 8,
    kwargs_imshow_ext=None,
    kwargs_scatter_matched=None,
    kwargs_scatter_lsst=None,
    kwargs_scatter_ext=None,
):
    if kwargs_imshow_ext is None:
        kwargs_imshow_ext = {}
    if kwargs_scatter_matched is None:
        kwargs_scatter_matched = (
            dict(s=100, edgecolor="darkred", marker="o", facecolor="none", label="Matched", ),
        )
    if kwargs_scatter_lsst is None:
        kwargs_scatter_lsst = (
            dict(s=40, edgecolor="lavender", marker="s", facecolor="none", label="LSST", ),
            dict(s=60, edgecolor="aquamarine", marker="s", facecolor="none", label="LSST", ),
        )

    if kwargs_scatter_ext is None:
        kwargs_scatter_ext = (
            dict(s=40, edgecolor="cornflowerblue", marker="D", facecolor="none", label="HST", ),
            dict(s=60, edgecolor="orange", marker="D", facecolor="none", label="HST", ),
        )

    good_ext = np.array(matched_in[id_ext]) >= 0
    good_comcam = np.array(matched_in["objectId"]) >= 0
    good_match = good_ext & good_comcam

    ra_lsst = matched_in["coord_ra"][good_comcam]
    dec_lsst = matched_in["coord_dec"][good_comcam]

    # TODO: Get more than i-band fluxes in the matched catalog
    flux_lsst = np.sum([matched_in[f"{band}_sersicFlux"] for band in ("i",)], axis=0)
    mag_lsst = (u.nJy * flux_lsst).to(u.ABmag).value

    flux_ext = np.sum([matched_in[flux] for flux in bands_fluxes_ext.values()], axis=0)
    mag_ext = (u.nJy * flux_ext).to(u.ABmag).value

    missing_bright_ext = (good_ext & ~good_comcam) & (mag_ext < 25)
    missing_bright_lsst = (~good_ext & good_comcam) & (mag_lsst < 25)

    fig_ext, ax_ext = plt.subplots(figsize=(2 * figsize_ax, 2 * figsize_ax), nrows=2)

    ax_ext[0].imshow(img_rgb_ext, extent=extent, **kwargs_imshow_ext)
    ax_ext[1].imshow(img_rgb_lsst, extent=extent)

    for axis in ax_ext:
        _scatter_multi(
            axis,
            matched_in["coord_best_ra"][good_match],
            matched_in["coord_best_dec"][good_match],
            kwargs_scatter_list=kwargs_scatter_matched,
        )
        for missing_bright, kwargs_bright in (
            (missing_bright_lsst, kwargs_scatter_lsst),
            (missing_bright_ext, kwargs_scatter_ext),
        ):
            _scatter_multi(
                axis,
                matched_in["coord_best_ra"][missing_bright],
                matched_in["coord_best_dec"][missing_bright],
                kwargs_scatter_list=kwargs_bright,
            )

    for idx_ell, (r_x, r_y, rho) in enumerate(zip(
        *(matched_in[f"sersic_{col}"][good_comcam] for col in ("reff_x", "reff_y", "rho"))
    )):
        ell_maj = g2d.EllipseMajor(g2d.Ellipse(r_x, r_y, rho), degrees=True)
        if ell_maj.r_major > 5:
            ell_patch = mpl.patches.Ellipse(
                xy=(ra_lsst[idx_ell], dec_lsst[idx_ell]),
                width=2*ell_maj.r_major*scale_lsst_deg,
                height=2*ell_maj.r_major*ell_maj.axrat*scale_lsst_deg,
                angle=-ell_maj.angle,
                edgecolor='gray', fc='None', lw=2,
            )
            ax_ext[1].add_artist(ell_patch)

    return fig_ext, ax_ext
