import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, pq
from lsst.sitcom.sciunit.galaxies.truth_summary import convert_truth_summary_v2_to_injection


truth_summary_path = "/sdf/data/rubin/shared/dc2_run2.2i_truth/truth_summary_cell"

is_hsc = False
plot = False

mag_total_min_star = 17.5
mag_total_min_galaxy = 15
mag_total_max = 26.5
n_skip = 1

tract_in = 3828

if is_hsc:
    bands = ("g", "r", "i", "z", "y")
    butler_out = dafButler.Butler("/repo/main", collections=["HSC/runs/RC2/w_2024_38/DM-46429"])
    skymap_name_out = "hsc_rings_v1"
    tract_out = 9813
else:
    bands = ("u", "g", "r", "i", "z", "y")
    butler_out = dafButler.Butler(
        "/repo/main",
        collections=["LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098"],
    )
    skymap_name_out = "lsst_cells_v1"
    tract_out = 5063

skymap_name_in = "DC2_cells_v1"

butler_in = dafButler.Butler("/repo/dc2")

catalogs, mags = convert_truth_summary_v2_to_injection(
    bands=bands,
    butler_in=butler_in,
    butler_out=butler_out,
    skymap_name_in=skymap_name_in,
    skymap_name_out=skymap_name_out,
    tract_in=tract_in,
    tract_out=tract_out,
    mag_total_min_star=mag_total_min_star,
    mag_total_min_galaxy=mag_total_min_galaxy,
    mag_total_max=mag_total_max,
    n_skip=n_skip,
)

for (filename, catalog), (band, mags_band) in zip(catalogs.items(), mags.items()):
    mag_column = catalog["mag"]
    desc = mag_column.description
    mag_column.description = f"{band}-band {desc}"
    mag_column[:] = mags_band
    pq.write_table(astropy_to_arrow(catalog), filename)
    mag_column.description = desc
