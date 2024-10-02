from lsst.daf.butler import Butler
from lsst.daf.butler.formatters.parquet import astropy_to_arrow
from lsst.geom import SpherePoint, degrees
import numpy as np
import pyarrow.parquet as pq

butler = Butler("/repo/dc2")
skymap_cell = butler.get("skyMap", skymap="DC2_cells_v1", collections="skymaps")
path = "/sdf/data/rubin/user/dtaranu/dc2/truth_summary_cell"

for tract in (3828, 3829):
    truth_summary = butler.get(
        "truth_summary", skymap="DC2", tract=tract, storageClass="ArrowAstropy",
        collections="2.2i/truth_summary"
    )
    radecs = [SpherePoint(ra, dec, degrees) for ra, dec in zip(truth_summary["ra"], truth_summary["dec"])]
    patches = [skymap_cell[tract].findPatch(radec).sequential_index for radec in radecs]
    truth_summary["patch"] = patches
    pq.write_table(
        astropy_to_arrow(truth_summary),
        f"{path}/truth_summary_{tract}_DC2_2_2i_truth_summary.parq",
    )
