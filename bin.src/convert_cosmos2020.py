import astropy.table as apTab
import astropy.units as u
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
import numpy as np
import pyarrow.parquet as pq

tab_ap = apTab.Table.read("COSMOS2020_CLASSIC_R1_v2.2_p3.fits")

unit_substitutes = {
    "log(solMass)": u.dex(u.solMass),
    "log(solMass/yr)": u.dex(u.solMass/u.yr),
    "log(yr**(-1))": u.dex(1.0/u.yr),
    "log(solLum)": u.dex(u.solLum),
    "uJy": "nJy",
}

for idx in range(len(tab_ap.columns)):
    column = tab_ap.columns[idx]
    unit_new = unit_substitutes.get(str(column.unit) if column.unit else None)
    msgs = [f"tab.columns[{idx}] {column.name}: "]
    if unit_new:
        msgs.append(f"changing {column.unit=} to {unit_new}")
        try:
            factor = u.Unit(column.unit).to(unit_new, 1.0)
            tab_ap[column.name] *= factor
            msgs.append(f"multiplying values by {factor=}")
        except Exception as exc:
            print(f"converting {unit_new=} got {exc=}")
        column.unit = unit_new
    dtype = column.dtype
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.unsignedinteger):
        if (np.nanmin(tab_ap[column.name]) == 0) and (np.nanmax(tab_ap[column.name]) == 1) and (
                dtype != bool):
            msgs.append(f"changing flag column {dtype=} to bool")

    if len(msgs) > 1:
        print(msgs[0] + "; ".join(msgs[1:]))

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, "COSMOS2020_CLASSIC_R1_v2.2_p3.parq")
