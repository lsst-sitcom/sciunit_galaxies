import astropy.units as u
from astropy.visualization import make_lupton_rgb
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
                              collections=["u/dtaranu/DM-47185/injected_dp1_2025_06/plots"])

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

objects = butler.get("objectTable_tract", skymap=skymap, tract=tract, storageClass="ArrowAstropy")
objects_patch = objects[objects["patch"] == patch]

objects_injected = butler.get("injected_objectTable_tract", skymap=skymap, tract=tract, storageClass="ArrowAstropy")
objects_injected_patch = objects_injected[objects_injected["patch"] == patch]

injected = butler.get("injected_deepCoadd_catalog_tract", skymap=skymap, tract=tract)
injected_patch = injected[injected["patch"] == patch]

matched = butler.get(
    "matched_injected_deepCoadd_catalog_tract_injected_objectTable_tract", skymap=skymap, tract=tract,
    storageClass="ArrowAstropy"
)

coadd = butler.get("deepCoadd_calexp", skymap=skymap, tract=tract, patch=patch, band=band)
coadd_injected = butler.get("injected_deepCoadd_calexp", skymap=skymap, tract=tract, patch=patch, band=band)

print("MASK [original, injected] (pixels)")
for mask in coadd.mask.getMaskPlaneDict():
    print(mask, [np.sum((c.mask.array & c.mask.getPlaneBitMask(mask)) > 0) for c in (coadd, coadd_injected)])

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

# TODO: This should already be the case but isn't - fix it
matched_patch[f"ref_{band}_flux"].unit = u.nJy

ref_mag = matched_patch[f"ref_{band}_flux"].to(u.ABmag).value
matched_bright = ref_mag < 24
matched_good = np.array(matched_patch["detect_isPrimary"] == True)

bright = injected_patch[f"{band}_mag"] < 24
flags = injected_patch[f"{band}_comp1_injection_flag"]

unique, counts = np.unique(flags, return_counts=True)
n_flags = len(unique)

colors = colormaps["plasma"].colors

def get_xys_from_radec(ra, dec):
    xys = np.array([
        [y for y in x] for x in coadd_injected.wcs.skyToPixel([
            SpherePoint(ra, dec, degrees)
            for ra, dec in zip(ra, dec)
        ])
    ])
    return xys

corners_xy = np.array(coadd_injected.getBBox().getCorners())
corners_radec = np.array([
    [y.asDegrees() for y in coadd_injected.wcs.pixelToSky(Point2D(x))]
    for x in coadd_injected.getBBox().getCorners()
])
extent = [
    np.mean(corners_xy[0:3:4, 0]), np.mean(corners_xy[1:3, 0]),
    np.mean(corners_xy[0:2, 1]), np.mean(corners_xy[2:4, 1]),
]

fig, ax = plt.subplots()
img_inj = coadd_injected.image.array
imgs_rgb = [
    ((coadd.mask.array & coadd.mask.getPlaneBitMask("DETECTED")) > 0).astype(int)*img_inj,
    img_inj,
    ((coadd_injected.mask.array & coadd_injected.mask.getPlaneBitMask("DETECTED")) > 0).astype(int)*img_inj,
]
ax.imshow(make_lupton_rgb(*imgs_rgb, Q=8, stretch=0.2), extent=extent)

xys = get_xys_from_radec(
    objects_injected_patch["coord_ra"][objects_injected_patch["detect_isPrimary"]],
    objects_injected_patch["coord_dec"][objects_injected_patch["detect_isPrimary"]],
)
ax.scatter(xys[:, 0], xys[:, 1], edgecolor='darkviolet', marker='o', label="detect_isPrimary", s=50, facecolor="None")

for idx, flag in enumerate(unique):
    color = colors[round((idx/(n_flags - 1 + (n_flags == 1)))*205 + 25)]
    to_plot = bright & (flags == flag)
    xys = get_xys_from_radec(injected_patch["ra"][to_plot], injected_patch["dec"][to_plot])
    ax.scatter(xys[:, 0], xys[:, 1], color=color, label=f"{flag=}", s=50)

xys = get_xys_from_radec(
    matched_patch["ref_ra"][matched_bright & matched_good],
    matched_patch["ref_dec"][matched_bright & matched_good],
)
ax.scatter(xys[:, 0], xys[:, 1], c='b', marker='+', label="good match", s=120)
xys = get_xys_from_radec(
    matched_patch["ref_ra"][matched_bright & ~matched_good],
    matched_patch["ref_dec"][matched_bright & ~matched_good],
)
ax.scatter(xys[:, 0], xys[:, 1], c='r', marker='x', label="no match", s=120)
ax.legend()
ax.set_title("RGB Red=coadd*detected, Green=injected, Blue=injected*detected")
fig.tight_layout()

fig, ax = plt.subplots()
img = coadd.image.array
imgs_rgb = [
    ((coadd.mask.array & coadd.mask.getPlaneBitMask("DETECTED")) > 0).astype(int)*img,
    img,
    ((coadd_injected.mask.array & coadd_injected.mask.getPlaneBitMask("DETECTED")) > 0).astype(int)*img,
]
ax.imshow(make_lupton_rgb(*imgs_rgb, Q=8, stretch=0.2), extent=extent)

xys = get_xys_from_radec(
    objects_patch["coord_ra"][objects_patch["detect_isPrimary"]],
    objects_patch["coord_dec"][objects_patch["detect_isPrimary"]],
)
ax.scatter(xys[:, 0], xys[:, 1], edgecolor='darkviolet', marker='o', label="detect_isPrimary", s=50, facecolor="None")
ax.legend()
ax.set_title("RGB Red=coadd*detected, Green=coadd, Blue=injected*detected")
fig.tight_layout()

plt.show()
