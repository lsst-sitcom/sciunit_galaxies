import astropy.io.ascii
import astropy.units as u
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
from lsst.geom import degrees, SpherePoint
import numpy as np
import pyarrow.parquet as pq

skymap = "lsst_cells_v1"
tract = 9813
butler = dafButler.Butler("/repo/main", collections="skymaps")
tractInfo = butler.get("skyMap", skymap=skymap)[tract]

tab_ap = astropy.io.ascii.read("cosmos_acs_iphot_200709.tbl")

# Leauthaud et al. 2007, ApJ says mags are AB
# Fluxes are in counts so not very useful except to rescale errors
flux_auto = u.ABmag.to(u.nJy, tab_ap["mag_auto"])
tab_ap["fluxerr_auto"] *= flux_auto/tab_ap["flux_auto"]
tab_ap["fluxerr_auto"].unit = u.nJy
tab_ap["flux_auto"] = flux_auto
tab_ap["flux_auto"].unit = u.nJy

coords = [
    SpherePoint(ra, dec, degrees) for ra, dec in zip(tab_ap["ra"], tab_ap["dec"])
]
within = np.array([tractInfo.contains(coord) for coord in coords])
if np.sum(within) != len(within):
    tab_ap = tab_ap[within]
    coords = [coord for coord, in_tract in zip(coords, within) if in_tract]
patches = np.array(
    [tractInfo.findPatch(coord).getSequentialIndex() for coord in coords],
    dtype=np.int16,
)
tab_ap["patch"] = patches
tab_ap["patch"].description = f"{skymap} patch index"

for column in ["ra", "dec"]:
    column_error = f"{column}_est_error"
    tab_ap[column_error] = np.full(len(tab_ap), 0.01 / 3600, dtype=np.float32)
    tab_ap[column_error].description = f"Placeholder {column_error} error (constant 10 mas)"
    tab_ap[column_error].unit = u.deg

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, "cosmos_acs_iphot_200709.parq", row_group_size=row_group_size)
