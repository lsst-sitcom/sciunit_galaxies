import math

import lsst.daf.butler as dafButler
import lsst.sphgeom as sphgeom
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from smatch.matcher import Matcher

mpl.rcParams.update({"image.origin": "lower", "font.size": 12, "figure.figsize": (16, 16)})

tract = 2396
butler = dafButler.Butler("dp2_prep", collections="u/dtaranu/DM-50135/DP2/matched_edfs")
skymap = "lsst_cells_v2"
skymapInfo = butler.get("skyMap", skymap=skymap, collections="skymaps")
tractInfo = skymapInfo[tract]

objects = butler.get(
    "object", skymap=skymap, tract=tract,
    parameters={"columns": ("objectId", "coord_ra", "coord_dec", "patch")}
)

ra, dec = objects["coord_ra"], objects["coord_dec"]

with Matcher(ra, dec) as matcher:
    idx, dists = matcher.query_knn(
        ra, dec,
        distance_upper_bound=1.0/3600., k=2, return_distances=True,
    )

idx_d1, idx_d2, crossp = {}, {}, {}
for idx, dist in enumerate(dists[:, 1]):
    if (dist > 0/3600) and (dist < 0.85/3600):
        idx_found = np.array(list(idx_d1.keys()))
        close = np.where(np.isclose(dist, np.array(list(idx_d1.values())), atol=1e-14, rtol=1e-14))[0]
        n_close = len(close)
        (idx_d1 if (n_close == 0) else idx_d2)[idx] = dist
        if n_close > 0:
            if objects["patch"][idx] != objects["patch"][idx_found[close[0]]]:
                crossp[idx] = int(idx_found[close[0]])

radec_dupe = 0.5*np.array([(ra[idx], dec[idx]) for idx in crossp.keys()])
radec_dupe += 0.5*np.array([(ra[idx], dec[idx]) for idx in crossp.values()])

fig, ax = plt.subplots()

sp = skyproj.GnomonicSkyproj(
    ax=ax, lon_0=np.nanmedian(ra), lat_0=np.nanmedian(dec),
    extent=(np.nanmin(ra), np.nanmax(ra), np.nanmin(dec), np.nanmax(dec)),
)

for patchInfo in tractInfo:
    vertices = patchInfo.getInnerSkyPolygon().getVertices()
    clipped = tractInfo.inner_sky_region.clipTo(
        sphgeom.Box(sphgeom.LonLat(vertices[0]), sphgeom.LonLat(vertices[2]))
    )
    lonlats = np.array([
        [x.asDegrees() for x in (lonlat.getA(), lonlat.getB())]
        for lonlat in (clipped.getLon(), clipped.getLat())
    ])
    lons = np.concat((lonlats[0, :], lonlats[0, ::-1]))
    lats = np.repeat(lonlats[1, :], 2)
    sp.draw_polygon(lons, lats, edgecolor="k")
    for cell in range(patchInfo.num_cells.x * patchInfo.num_cells.y):
        cellInfo = patchInfo.getCellInfo(cell)
        vertices = cellInfo.getInnerSkyPolygon().getVertices()
        clipped = tractInfo.inner_sky_region.clipTo(
            sphgeom.Box(sphgeom.LonLat(vertices[0]), sphgeom.LonLat(vertices[2]))
        )
        lonlats = np.array([
            [x.asDegrees() for x in (lonlat.getA(), lonlat.getB())]
            for lonlat in (clipped.getLon(), clipped.getLat())
        ])
        if not np.isfinite(lonlats).all():
            continue
        lons = np.concat((lonlats[0, :], lonlats[0, ::-1]))
        lats = np.repeat(lonlats[1, :], 2)
        sp.draw_polygon(lons, lats, edgecolor="k", linewidth=200/3600)

sp.ax.scatter(ra[list(idx_d1.keys())], dec[list(idx_d1.keys())], s=100, marker="x", c="b")
sp.ax.scatter(ra[list(idx_d2.keys())], dec[list(idx_d2.keys())], s=130, marker="+", c="r")
sp.ax.scatter(radec_dupe[:, 0], radec_dupe[:, 1], s=150, marker="o", edgecolor="purple", facecolor="None")
fig.tight_layout()
plt.show()

