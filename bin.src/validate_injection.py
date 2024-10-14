from lsst.daf.butler.formatters.parquet import arrow_to_astropy, pq
import lsst.daf.butler as dafButler
import lsst.geom
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["image.origin"] = "lower"

tract = 9813
patch = 40
band = "i"
cutout_asec = 60

repo = "/repo/main"
butler = dafButler.Butler(repo, skymap="hsc_rings_v1", collections=["u/dtaranu/DM-44943/injected"])
injection = arrow_to_astropy(pq.read_table("injection_catalog_hsc_rings_v1_9813_from_DC2_cells_v1_r_3828.parq"))
butler_dc2 = dafButler.Butler(
    "/repo/dc2", skymap="DC2_cells_v1", collections=["2.2i/runs/test-med-1/w_2024_40/DM-46604"],
)
calexp_hsc = butler.get("deepCoadd_calexp", patch=patch, tract=tract, band=band)
calexp_inj = butler.get("injected_deepCoadd", patch=patch, tract=tract, band=band)
truth_summary_v2 = arrow_to_astropy(pq.read_table(
    "/sdf/data/rubin/user/dtaranu/dc2/truth_summary_cell/truth_summary_v2_3828_DC2_cells_v1_2_2i_truth_summary.parq"
))

objects = butler.get(
    "objectTable_tract", tract=tract, storageClass="ArrowAstropy",
    parameters={"columns": ("coord_ra", "coord_dec", "detect_isPrimary",)},
)

primary = objects[objects["detect_isPrimary"]]
plt.scatter(objects["coord_ra"][::20], objects["coord_dec"][::20], s=0.8)
plt.scatter(injection["ra"][::20], injection["dec"][::20], s=0.8)
plt.tight_layout()
plt.show()

pa_offset = 90

ids_ref = [
    7813120317, 7812500182, 7812541653, 7812608827, 7812509574,
    7812509636, 7812612080, 7812647026, 7812647658, 7813026314,
]
for id_ref in ids_ref:
    obj_ref = injection[injection["group_id"] == id_ref][0]
    ra_hsc, dec_hsc = (obj_ref[c] for c in ("ra", "dec"))

    obj_dc2 = truth_summary_v2[truth_summary_v2["id"] == id_ref][0]
    calexp_dc2 = butler_dc2.get("deepCoadd_calexp", patch=obj_dc2["patch"], tract=obj_dc2["tract"], band=band)

    cutout_hsc, cutout_hsc_inj = (
        calexp.getCutout(
            center=lsst.geom.SpherePoint(ra_hsc, dec_hsc, lsst.geom.degrees),
            size=lsst.geom.Extent2I(int(cutout_asec/0.168), int(cutout_asec/0.168)),
        )
        for calexp in (calexp_hsc, calexp_inj)
    )

    coord_dc2 = lsst.geom.SpherePoint(obj_dc2["ra"], obj_dc2["dec"], lsst.geom.degrees)

    cutout_dc2 = calexp_dc2.getCutout(
        center=coord_dc2,
        size=lsst.geom.Extent2I(int(cutout_asec / 0.2), int(cutout_asec / 0.2)),
    )

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

    min_in = np.nanmin(cutout_hsc.image.array)
    max_in = np.nanmax(cutout_hsc.image.array)

    pa = obj_dc2['positionAngle']

    ax[0][0].imshow(np.arcsinh(cutout_hsc.image.array*10), cmap="gray")
    ax[0][0].set_title(f"HSC {tract=} {patch=} {band=}")
    ax[0][1].imshow(np.arcsinh(np.clip(cutout_hsc_inj.image.array, min_in, max_in)*10), cmap="gray")
    ax[0][1].set_title(f"HSC injected {tract=} {patch=} {band=}")
    ax[1][0].imshow(
        np.arcsinh(10*np.clip(cutout_hsc_inj.image.array - cutout_hsc.image.array, min_in, max_in)),
        cmap="gray",
    )
    ax[1][0].set_title(f"injected difference")
    ax[1][1].imshow(np.arcsinh(np.clip(cutout_dc2.image.array, min_in, max_in)*10), cmap="gray")
    ax[1][1].set_title(f"DC2 id={id_ref} tract={obj_dc2['tract']} patch={obj_dc2['patch']}"
                       f" {pa_offset}-PA={pa_offset - pa:.1f}")
    coord = [x/2 for x in cutout_dc2.image.array.shape[::-1]]
    linelen = coord[0]/8
    dx = np.cos((-pa + pa_offset)*np.pi/180)
    dy = np.sin((-pa + pa_offset)*np.pi/180)
    ax[1][1].plot([coord[0]-linelen*dx, coord[0]+linelen*dx], [coord[1]-linelen*dy, coord[1]+linelen*dy], 'r-')

    plt.tight_layout()

plt.show()
