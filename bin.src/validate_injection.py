# Validate a DC2-based injection run
# Should be in ComCam; HSC may not work as well

import astropy.units as u
from astropy.visualization import make_lupton_rgb
import lsst.afw.image
import lsst.daf.butler as dafButler
from lsst.geom import Point2D, SpherePoint, degrees
from lsst.sitcom.sciunit.galaxies.truth_summary import validate_injection_catalog
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"image.origin": "lower", 'font.size': 18, "figure.figsize": (15, 15)})

is_rc2 = False
plot_validation = False
cutout_asec = 60
tract_in = 3828
repo_dc2 = "/repo/dc2"
skymap_dc2 = "DC2_cells_v1"
bands = ("i", "r", "g")

if is_rc2:
    tract = 9813
    patch = 40
    band = "i"

    repo = "/repo/main"
    skymap = "hsc_rings_v1"
    butler = dafButler.Butler(repo, skymap=skymap, collections=["u/dtaranu/DM-44943/injected"])
else:
    tract = 5063
    patch = 33
    band = "r"
    repo = "/repo/main"
    skymap = "lsst_cells_v1"
    butler = dafButler.Butler(repo, skymap=skymap,
                              collections=["u/dtaranu/DM-50425/injected_dp1_v29_0_0_rc6/plots"])


def calibrate_exposure(exposure: lsst.afw.image.Exposure) -> lsst.afw.image.Exposure:
    calib = exposure.getPhotoCalib()
    image = calib.calibrateImage(
        lsst.afw.image.MaskedImageF(exposure.image, mask=exposure.mask, variance=exposure.variance)
    )
    exposure.image.array[:] = image.image.array[:]
    exposure.mask.array[:] = image.mask.array[:]
    exposure.variance.array[:] = image.variance.array[:]
    return exposure

butler_dc2 = dafButler.Butler(
    repo_dc2, skymap=skymap_dc2, collections=["2.2i/runs/test-med-1/w_2024_40/DM-46604"],
)

if plot_validation:
    validate_injection_catalog(
        butler_in=butler_dc2,
        butler_out=butler,
        tract_in=tract_in,
        tract_out=tract,
        skymap_name_in=skymap_dc2,
        skymap_name_out=skymap,
        patch=patch,
        band=band,
        cutout_asec=cutout_asec,
        ids_ref=[],
    )

coadd_data = calibrate_exposure(
    butler.get("deep_coadd", skymap=skymap, tract=tract, patch=patch, band=band)
)
# HSC skymaps aren't compatible with cells (and probably won't be made so)
coadd_dc2 = None if is_rc2 else calibrate_exposure(
    butler_dc2.get("deepCoadd_calexp", skymap=skymap_dc2, tract=tract_in, patch=patch, band=band)
)
coadd_injected = calibrate_exposure(
    butler.get("injected_deep_coadd", skymap=skymap, tract=tract, patch=patch, band=band)
)

columns = ("coord_ra", "coord_dec", "patch",)
columns_old = ("detect_isPrimary",)

coadds = {}
objects = {}
for subtype, butler_sub, skymap_sub, tract_sub, patch_sub, prefix, old in (
    ("Injected", butler, skymap, tract, patch, "injected_", False),
    ("DC2", butler_dc2, skymap_dc2, tract_in, patch, "", True),
    ("Data", butler, skymap, tract, None if is_rc2 else patch, "", False),
):
    objects_sub = butler_sub.get(
        f"{prefix}{'objectTable_tract' if old else 'object'}",
        skymap=skymap_sub, tract=tract_sub, storageClass="ArrowAstropy",
        parameters={"columns": columns + (columns_old if old else tuple())},
    )
    if not old:
        objects_sub["detect_isPrimary"] = True
    objects[subtype] = objects_sub[objects_sub["patch"] == patch]
    coadds[subtype] = butler_sub.get(
        f"{prefix}{'deepCoadd_calexp' if old else 'deep_coadd'}",
        skymap=skymap_sub, tract=tract_sub, patch=patch_sub, band=band,
    )

columns_injected = tuple(
    name.format(band=band) for name in ("{band}_mag", "{band}_comp1_injection_flag") for band in bands
) + ("patch", "ra", "dec")

injected = butler.get(
    "injected_deepCoadd_catalog_tract" if is_rc2 else "injected_deep_coadd_predetection_catalog_tract",
    skymap=skymap, tract=tract, storageClass="ArrowAstropy",
    parameters={"columns": columns_injected},
)
injected = injected[injected["patch"] == patch]
injected.rename_columns(("ra", "dec"), ("coord_ra", "coord_dec"))

columns_matched = tuple(
    name.format(band=band)
    for name in (
        "ref_{band}_flux", "{band}_cModelFlux", "{band}_kronFlux", "{band}_gaap1p0Flux", "{band}_gaap3p0Flux",
    )
    for band in bands
) + (
    "coord_ra", "coord_dec", "detect_isPrimary", "objectId", "patch", "ref_injected_id", "ref_ra", "ref_dec"
)

matched = butler.get(
    ("matched_injected_deepCoadd_catalog_tract_injected_objectTable_tract" if is_rc2 else
     "matched_injected_deep_coadd_predetection_catalog_tract_injected_object_all"),
    skymap=skymap, tract=tract, storageClass="ArrowAstropy", parameters={"columns": columns_matched},
)

tractInfo = butler.get("skyMap", skymap=skymap)[tract]
no_patch = matched["patch"].mask | ~(matched["patch"] >= 0).data
matched["patch"].mask = False
new_patches = np.full(np.sum(no_patch), -1, dtype=matched["patch"].dtype)
for idx, row in enumerate(matched[no_patch]):
    new_patches[idx] = tractInfo.findPatch(
        SpherePoint(row["ref_ra"], row["ref_dec"], degrees)
    ).sequential_index
matched["patch"][no_patch] = new_patches
matched_patch = matched[matched["patch"] == patch]

flux_matched = []
flux_injected = []

for band in bands:
    # TODO: This should already be the case but isn't - fix it
    matched_patch[f"ref_{band}_flux"].unit = u.nJy
    flux_matched.append(matched_patch[f"ref_{band}_flux"])
    flux_injected.append(injected["r_mag"].to(u.nJy).value)

mag_matched = np.nansum(flux_matched, axis=0)
mag_ref = np.nansum(flux_injected, axis=0)

matched_bright = mag_ref < 23
matched_good = np.array(matched_patch["detect_isPrimary"] == True)

bright = injected[f"{band}_mag"] < 24
flags = injected[f"{band}_comp1_injection_flag"]

unique, counts = np.unique(flags, return_counts=True)
n_flags = len(unique)

colors = colormaps["plasma"].colors

def get_xys_from_radec(ra, dec, coadd):
    xys = np.array([
        [y for y in x] for x in coadd.wcs.skyToPixel([
            SpherePoint(ra, dec, degrees)
            for ra, dec in zip(ra, dec)
        ])
    ])
    return xys


kwargs_primary = dict(edgecolor='cornflowerblue', marker='o', s=50, facecolor="None")
kwargs_primary_injected = dict(facecolor='orange', marker='+', s=50)
kwargs_primary_data = dict(edgecolor='limegreen', marker='o', s=50, facecolor="None")

xlim = (9000, 9500)
ylim = (9000, 9500)

detected_injected = coadd_injected.mask.getPlaneBitMask("DETECTED")
is_detected_injected = ((coadd_injected.mask.array & detected_injected) > 0).astype(int)

for subtype, coadd in (("Injected", coadd_injected), ("DC2", coadd_dc2), ("Data", coadd_data)):
    if coadd is not None:
        is_injected = subtype == "Injected"
        objects_sub = objects[subtype]
        print(f"{subtype} mask statistics:")
        for mask in coadd.mask.getMaskPlaneDict():
            print(mask, [np.sum((c.mask.array & c.mask.getPlaneBitMask(mask)) > 0) for c in
                         (coadd_dc2, coadd_injected)])

        fig, ax = plt.subplots()

        corners_xy = np.array(coadd.getBBox().getCorners())
        corners_radec = np.array([
            [y.asDegrees() for y in coadd.wcs.pixelToSky(Point2D(x))]
            for x in coadd.getBBox().getCorners()
        ])
        extent = [
            np.mean(corners_xy[0:3:4, 0]), np.mean(corners_xy[1:3, 0]),
            np.mean(corners_xy[0:2, 1]), np.mean(corners_xy[2:4, 1]),
        ]

        img = coadd.image.array
        imgs_rgb = [
            ((coadd.mask.array & coadd.mask.getPlaneBitMask("DETECTED")) > 0).astype(int)*img,
            img,
            is_detected_injected*img,
        ]
        ax.imshow(make_lupton_rgb(*imgs_rgb, Q=8, stretch=10), extent=extent)

        to_plot = {
            "primary": (objects_sub, kwargs_primary),
        }
        if is_injected:
            to_plot["injected"] = (injected, kwargs_primary_injected)
            to_plot["data"] = (objects["Data"], kwargs_primary_data)

        for label, (objects_plot, kwargs_plot) in to_plot.items():
            ra, dec = (objects_plot[f"coord_{c}"] for c in ("ra", "dec"))
            xys = get_xys_from_radec(
                ra[objects_plot["detect_isPrimary"]] if is_rc2 else ra,
                dec[objects_plot["detect_isPrimary"]] if is_rc2 else dec,
                coadd
            )
            ax.scatter(xys[:, 0], xys[:, 1], label=label, **kwargs_plot)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend()
        ax.set_title(f"{subtype} RGB Red=coadd*detected, Green=coadd, Blue=injected*detected")
        fig.tight_layout()

plt.show()
