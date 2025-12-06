# Script to inspect CDFS objects/images and classify them
# Makes interactive matplotlib figures
# This exercise didn't work as well as I hoped

import lsst.daf.butler as dafButler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from lsst.sitcom.sciunit.galaxies.cdfs_hst import get_cutouts_cdfs_hst
from lsst.sitcom.sciunit.galaxies.euclid import get_cutouts_euclid

mpl.rcParams.update({"image.origin": "lower", "font.size": 12, "figure.figsize": (12, 12)})

cmap = "gray"

interactive = True
if interactive:
    from matplotlib.widgets import Button

    class Index:
        class_ambiguous = 0.5
        class_bad = -1.
        class_galaxy = 0.
        class_nothing = -99.
        class_star = 1.
        class_blend = 2.

        ind = 0
        values = {}

        def ambiguous(self, event):
            self.values[self.ind] = self.class_ambiguous
            plt.close()

        def bad(self, event):
            self.values[self.ind] = self.class_bad
            plt.close()

        def blend(self, event):
            self.values[self.ind] = self.class_blend
            plt.close()

        def galaxy(self, event):
            self.values[self.ind] = self.class_galaxy
            plt.close()

        def nothing(self, event):
            self.values[self.ind] = self.class_nothing
            plt.close()

        def star(self, event):
            self.values[self.ind] = self.class_star
            plt.close()

        @property
        def class_callbacks(self):
            return {
                "Nothing": self.nothing,
                "Artifact/Bad": self.bad,
                "Galaxy": self.galaxy,
                "Ambiguous": self.ambiguous,
                "Star": self.star,
                "Blend": self.blend,
            }

    callback = Index()

vertical = True
ivert = int(vertical)
nivert = 1 - ivert

collection = "u/dtaranu/DM-50135/w_2025_39/matched_cdfs"
butler = dafButler.Butler("/repo/main", collections=collection)

skymap = "lsst_cells_v1"
skymapInfo = butler.get("skyMap", skymap=skymap, collections="skymaps")

tract = 5063

matched = butler.get("matched_cdfs_mast_euclid_q1_object", skymap=skymap, tract=tract, collections=collection)

u_ra = matched["coord_best_ra"].unit
u_dec = matched["coord_best_dec"].unit

bands_euclid = ("VIS", "Y")
bands_hst = ("F435W", "F606W", "F775W", "F814W")

for band in bands_euclid:
    matched[f"mag_{band}"] = -2.5*np.log10(matched[f"euclid_flux_{band.lower()}_sersic"]) + 31.4

for band in bands_hst:
    matched[f"mag_{band}"] = -2.5*np.log10(matched[f"hst_f_{band.lower()}"]) + 31.4

cutout_size_hst = (120, 120)
cutout_size_euclid = (40, 40)

for patch in (26,):
    if interactive:
        callback.values = {}
    matched_patch = matched[matched["hst_patch"] == patch]
    ambig_patch = matched_patch[
        (np.abs(matched_patch["hst_class_star"] - matched_patch["euclid_point_like_prob"]) > 0.5) == True
    ]

    fits_cdfs = {}
    fits_euclid = {}

    for ambig in ambig_patch:
        coord = SkyCoord(ambig["coord_best_ra"]*u_ra, ambig["coord_best_dec"]*u_dec)
        cutouts, extent_hst = get_cutouts_cdfs_hst(
            tract, patch, bands_hst, skymap, coord, cutout_size_hst, fits_cdfs=fits_cdfs, keep_fits=True,
        )
        extents = {band: extent_hst for band in cutouts.keys()}
        cutouts_euclid, extent_euclid = get_cutouts_euclid(
            skymap, tract, patch, bands_euclid, coord, cutout_size_euclid,
            fits_euclid=fits_euclid, keep_fits=True,
        )
        for band, cutout in cutouts_euclid.items():
            extents[band] = extent_euclid
            cutouts[band] = cutout

        fig, ax = plt.subplots(2 + ivert, 2 + nivert)
        fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95)

        for band, i, j in (
            ("F435W", 0, 0),
            ("F606W", nivert, ivert),
            ("F775W", ivert, nivert),
            ("F814W", 1, 1),
            ("VIS", 2*ivert, 2 * nivert),
            ("Y", 2*ivert + nivert, 2 * nivert + ivert)
        ):
            axis = ax[i][j]
            axis.imshow(np.arcsinh(cutouts[band].data), extent=extents[band], cmap=cmap)
            mag = ambig[f"mag_{band}"]
            axis.set_title(f"{band} {mag:.2f} mag")

        fig.suptitle(
            f"hst_id={ambig['hst_id']} class_star={ambig['hst_class_star']:.2f}"
            f" euclid_pp={ambig['euclid_point_like_prob']:.2f}"
        )
        if interactive:
            callback.ind = int(ambig["hst_id"])
            x = 0.15
            buttons = {}
            for label, class_callback in callback.class_callbacks.items():
                ax_button = fig.add_axes([x, 0.01, 0.1, 0.05])
                button = Button(ax_button, label)
                button.on_clicked(class_callback)
                buttons[label] = button
                x += 0.15

        plt.show()

    if interactive:
        output = Table({
            "hst_id": np.array(list(callback.values.keys())),
            "class_eye": np.array(list(callback.values.values()))}
        )
        output.write(f"cdfs_classes_{patch}.ecsv")
