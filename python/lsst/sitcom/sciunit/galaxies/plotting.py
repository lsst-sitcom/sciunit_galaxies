from typing import Any

import astropy.units as u
import lsst.gauss2d as g2d
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .lsst import scale_lsst_deg


def _scatter_multi(axis, x, y, kwargs_scatter_list):
    for kwargs_scatter in kwargs_scatter_list:
        axis.scatter(x, y, **kwargs_scatter)


def plot_external_matches(
    img_rgb_lsst,
    img_rgb_ext,
    matched_in,
    good_ext: NDArray[bool],
    good_lsst: NDArray[bool],
    detectable_ext: NDArray[bool],
    detectable_lsst: NDArray[bool],
    key_ra_ref="coord_best_ra",
    key_dec_ref="coord_best_dec",
    label_ext_default="HST",
    kwargs_imshow_ext: dict[str, Any] = None,
    kwargs_imshow_lsst: dict[str, Any] = None,
    kwargs_scatter_matched: dict[str, Any] = None,
    kwargs_scatter_lsst: dict[str, Any] = None,
    kwargs_scatter_ext: dict[str, Any] = None,
    kwargs_subplots: dict[str, Any] = None,
):
    """

    Parameters
    ----------
    img_rgb_lsst
        The LSST RGB image.
    img_rgb_ext
        An external RGB image covering the same area.
    matched_in
        The matched catalog subset within the image extent.
    good_ext
        A boolean array selecting matched_in rows with good external measurements.
    good_lsst
        A boolean array selecting matched_in rows with good LSST measurements.
    detectable_ext
        A boolean array selecting matched_in rows with external fluxes bright
        enough to be detectable in LSST. Missing
    detectable_lsst
        A boolean array selecting matched_in rows with LSST fluxes bright
        enough to be detectable in the external dataset.
    key_ra_ref
        The external right ascension column name.
    key_dec_ref
        The external declination column name.
    label_ext_default
        The label or name of the external dataset/telescope.
    kwargs_imshow_ext
        Keyword arguments to pass to plt.imshow for the external image.
    kwargs_imshow_lsst
        Keyword arguments to pass to plt.imshow for the LSST image.
    kwargs_scatter_matched
        Keyword arguments to pass to plt.scatter for the matched objects.
    kwargs_scatter_lsst
        Keyword arguments to pass to plt.scatter for the unmatched LSST objects.
    kwargs_scatter_ext
        Keyword arguments to pass to plt.scatter for the unmatched external objects.
    kwargs_subplots
        Keyword arguments to pass to plt.subplots

    Returns
    -------
    fig_ext, ax_ext
        The matplotlib Figure and Axes.
    """
    if kwargs_imshow_ext is None:
        kwargs_imshow_ext = {}
    if kwargs_imshow_lsst is None:
        kwargs_imshow_lsst = {}
    if kwargs_scatter_matched is None:
        kwargs_scatter_matched = (
            dict(s=100, edgecolor="darkred", marker="o", facecolor="none", label="Matched", ),
        )
    if kwargs_scatter_lsst is None:
        kwargs_scatter_lsst = (
            dict(s=40, edgecolor="lavender", marker="s", facecolor="none", label="LSST only"),
            dict(s=60, edgecolor="aquamarine", marker="s", facecolor="none"),
        )
    if kwargs_scatter_ext is None:
        kwargs_scatter_ext = (
            dict(s=40, edgecolor="cornflowerblue", marker="D", facecolor="none"),
            dict(s=60, edgecolor="orange", marker="D", facecolor="none", label=f"{label_ext_default} only"),
        )
    if kwargs_subplots is None:
        kwargs_subplots = {}

    good_match = good_ext & good_lsst

    ra_lsst = matched_in["coord_ra"][good_lsst]
    dec_lsst = matched_in["coord_dec"][good_lsst]

    unmatched_bright_ext = (good_ext & ~good_lsst) & detectable_ext
    unmatched_bright_lsst = (~good_ext & good_lsst) & detectable_lsst

    fig_ext, ax_ext = plt.subplots(nrows=2, ncols=1, **kwargs_subplots)

    ax_ext[0].imshow(img_rgb_ext, **kwargs_imshow_ext)
    ax_ext[1].imshow(img_rgb_lsst, **kwargs_imshow_lsst)

    for axis in ax_ext:
        _scatter_multi(
            axis,
            matched_in[key_ra_ref][good_match],
            matched_in[key_dec_ref][good_match],
            kwargs_scatter_list=kwargs_scatter_matched,
        )
        for unmatched_bright, kwargs_bright in (
            (unmatched_bright_lsst, kwargs_scatter_lsst),
            (unmatched_bright_ext, kwargs_scatter_ext),
        ):
            _scatter_multi(
                axis,
                matched_in[key_ra_ref][unmatched_bright],
                matched_in[key_dec_ref][unmatched_bright],
                kwargs_scatter_list=kwargs_bright,
            )

    for idx_ell, (r_x, r_y, rho) in enumerate(zip(
        *(matched_in[f"sersic_{col}"][good_lsst] for col in ("reff_x", "reff_y", "rho"))
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
