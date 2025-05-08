import astropy.io.ascii
import astropy.units as u
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
import pyarrow.parquet as pq

tab_ap = astropy.io.ascii.read("cosmos_acs_iphot_200709.tbl")

# Leauthaud et al. 2007, ApJ says mags are AB
# Fluxes are in counts so not very useful except to rescale errors
flux_auto = u.ABmag.to(u.nJy, tab_ap["mag_auto"])
tab_ap["fluxerr_auto"] *= flux_auto/tab_ap["flux_auto"]
tab_ap["fluxerr_auto"].unit = u.nJy
tab_ap["flux_auto"] = flux_auto
tab_ap["flux_auto"].unit = u.nJy

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, "cosmos_acs_iphot_200709.parq", row_group_size=row_group_size)


def register():
    import lsst.daf.butler as dafButler
    butler = dafButler.Butler("/repo/main", writeable=True)
    butler.registry.registerRun("u/dtaranu/DM-44000/cosmos_acs")
