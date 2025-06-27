from astropy.table import Table

from lsst.utils.plotting.figures import get_multiband_plot_colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "image.origin": "lower", "font.size": 13, "figure.figsize": (18, 18),
})
colors = get_multiband_plot_colors()
scale = 0.2
sma_max_asec = 1.0

for obj, rebin in (("M49", True), ("NGC4261", False)):
    prefix = "rebin_" if rebin else ""
    # intensities are averages so surface brightness must use scale
    scale_rebin = scale*(1 + rebin)
    tabs = {}
    for band in "irg":
        tab = Table.read(f"{obj.lower()}_ellipse_resid_asinh_{prefix}ellipses_{band}.ecsv")
        tab = tab[tab["sma"]*scale_rebin > sma_max_asec]
        tabs[band] = tab

    fig, ax = plt.subplots()
    for band, tab in tabs.items():
        ax.plot(np.log10(tab["sma"]*scale_rebin), -2.5*np.log10(tab["intens"]/(scale**2)) + 31.4,
                label=band, c=colors[band])
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(r"log10(Rmaj/asec)")
    ax.set_ylabel(r"μ (mag/asec^2)")
    ax.set_title(obj)
    fig.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    for band, tab in tabs.items():
        ax.plot((tab["sma"]*scale_rebin)**0.25, -2.5*np.log10(tab["intens"]/(scale**2)) + 31.4,
                label=band, c=colors[band])
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(r"(Rmaj/asec)^(1/4)")
    ax.set_ylabel(r"μ (mag/asec^2)")
    ax.set_title(obj)
    fig.legend()
    fig.tight_layout()

plt.show()

