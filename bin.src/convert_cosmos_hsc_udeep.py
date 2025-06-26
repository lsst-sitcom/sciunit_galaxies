import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, astropy_to_arrow, compute_row_group_size
from lsst.geom import degrees, SpherePoint
import numpy as np
import pyarrow.parquet as pq

skymap = "lsst_cells_v1"
tract = 9813
name_tab = f"rc2_object_hsc_{skymap}_{tract}"
butler = dafButler.Butler("/repo/main", collections="skymaps")
tractInfo = butler.get("skyMap", skymap=skymap)[tract]

# Or you can butler get it
tab_ap = arrow_to_astropy(pq.read_table("objectTable_tract_9813_hsc_rings_v1.parq"))
tab_ap = tab_ap[tab_ap["detect_isPrimary"] == True]
del tab_ap["detect_isPrimary"]
del tab_ap["merge_peak_sky"]

coords = [
    SpherePoint(ra, dec, degrees) for ra, dec in zip(tab_ap["coord_ra"], tab_ap["coord_dec"])
]
within = np.array([tractInfo.contains(coord) for coord in coords])
if np.sum(within) != len(within):
    tab_ap = tab_ap[within]
    coords = [coord for coord, in_tract in zip(coords, within) if in_tract]
patches = np.array(
    [tractInfo.findPatch(coord).getSequentialIndex() for coord in coords],
    dtype=np.int16,
)
tab_ap["patch_original"] = tab_ap["patch"]
tab_ap["patch"] = patches

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, f"{name_tab}.parq", row_group_size=row_group_size)
