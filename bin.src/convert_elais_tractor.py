from itertools import compress

import astropy.table as apTab
import astropy.units as u
from lsst.afw.image import fluxErrFromABMagErr
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
from lsst.geom import degrees, SpherePoint
import numpy as np
import pyarrow.parquet as pq

skymap = "lsst_cells_v1"
name_tab = "ES1phot"
butler = dafButler.Butler("/repo/main", collections="skymaps")
skymapInfo = butler.get("skyMap", skymap=skymap)

tab_ap = apTab.Table.read(f"{name_tab}.fits")

columns = {
    "ID_VIDEO": "VIDEO survey ID",
    "RA": ("Right ascension", u.arcsec),
    "Dec": ("Declination", u.arcsec),
    "FiducialBand": "Fiducial VIDEO band for fitting",
    "SourceModel": "Type of surface brightness profile used in Tractor fitting",
    "mag_ch2_DeepDrill": "AB ch2-band magnitude in DeepDrill",
    "mag_ch1_DeepDrill": "AB ch1-band magnitude in DeepDrill",
    "mag_Ks_VIDEO": "AB Ks-band magnitude in VIDEO",
    "mag_H_VIDEO": "AB H-band magnitude in VIDEO",
    "mag_J_VIDEO": "AB J-band magnitude in VIDEO",
    "mag_Y_VIDEO": "AB Y-band magnitude in VIDEO",
    "mag_Z_VIDEO": "AB Z-band magnitude in VIDEO",
    "mag_y_DES": "AB y-band magnitude in DES",
    "mag_z_DES": "AB z-band magnitude in DES",
    "mag_i_DES": "AB i-band magnitude in DES",
    "mag_r_DES": "AB r-band magnitude in DES",
    "mag_g_DES": "AB g-band magnitude in DES",
    "mag_R_ESIS": "AB R-band magnitude in ESIS",
    "mag_V_ESIS": "AB V-band magnitude in ESIS",
    "mag_B_ESIS": "AB B-band magnitude in ESIS",
    "mag_u_VOICE": "AB u-band magnitude in VOICE",
    "magerr_ch2_DeepDrill": "AB ch2-band magnitude error in DeepDrill",
    "magerr_ch1_DeepDrill": "AB ch1-band magnitude error in DeepDrill",
    "magerr_Ks_VIDEO": "AB Ks-band magnitude error in VIDEO",
    "magerr_H_VIDEO": "AB H-band magnitude error in VIDEO",
    "magerr_J_VIDEO": "AB J-band magnitude error in VIDEO",
    "magerr_Y_VIDEO": "AB Y-band magnitude error in VIDEO",
    "magerr_Z_VIDEO": "AB Z-band magnitude error in VIDEO",
    "magerr_y_DES": "AB y-band magnitude error in DES",
    "magerr_z_DES": "AB z-band magnitude error in DES",
    "magerr_i_DES": "AB i-band magnitude error in DES",
    "magerr_r_DES": "AB r-band magnitude error in DES",
    "magerr_g_DES": "AB g-band magnitude error in DES",
    "magerr_R_ESIS": "AB R-band magnitude error in ESIS",
    "magerr_V_ESIS": "AB V-band magnitude error in ESIS",
    "magerr_B_ESIS": "AB B-band magnitude error in ESIS",
    "magerr_u_VOICE": "AB u-band magnitude error in VOICE",
    "redchi2_ch2_DeepDrill": "ch2-band reduced chi-squared in DeepDrill",
    "redchi2_ch1_DeepDrill": "ch1-band reduced chi-squared in DeepDrill",
    "redchi2_Ks_VIDEO": "Ks-band reduced chi-squared in VIDEO",
    "redchi2_H_VIDEO": "H-band reduced chi-squared in VIDEO",
    "redchi2_J_VIDEO": "J-band reduced chi-squared in VIDEO",
    "redchi2_Y_VIDEO": "Y-band reduced chi-squared in VIDEO",
    "redchi2_Z_VIDEO": "Z-band reduced chi-squared in VIDEO",
    "redchi2_y_DES": "y-band reduced chi-squared in DES",
    "redchi2_z_DES": "z-band reduced chi-squared in DES",
    "redchi2_i_DES": "i-band reduced chi-squared in DES",
    "redchi2_r_DES": "r-band reduced chi-squared in DES",
    "redchi2_g_DES": "g-band reduced chi-squared in DES",
    "redchi2_R_ESIS": "R-band reduced chi-squared in ESIS",
    "redchi2_V_ESIS": "V-band reduced chi-squared in ESIS",
    "redchi2_B_ESIS": "B-band reduced chi-squared in ESIS",
    "redchi2_u_VOICE": "u-band reduced chi-squared in VOICE",
    "flag_sat_ch2_DeepDrill": "ch2-band saturated flag in DeepDrill",
    "flag_sat_ch1_DeepDrill": "ch1-band saturated flag in DeepDrill",
    "flag_sat_Ks_VIDEO": "Ks-band saturated flag in VIDEO",
    "flag_sat_H_VIDEO": "H-band saturated flag in VIDEO",
    "flag_sat_J_VIDEO": "J-band saturated flag in VIDEO",
    "flag_sat_Y_VIDEO": "Y-band saturated flag in VIDEO",
    "flag_sat_Z_VIDEO": "Z-band saturated flag in VIDEO",
    "flag_sat_R_ESIS": "R-band saturated flag in ESIS",
    "flag_sat_V_ESIS": "V-band saturated flag in ESIS",
    "flag_sat_B_ESIS": "B-band saturated flag in ESIS",
    "flag_outlier_DES": "DES outlier flag",
    "nndist": ("nearest-neighbor angular distance", u.arcsec),
}

rename = {}
for name_column, desc in columns.items():
    column = tab_ap.columns[name_column]
    if isinstance(desc, str):
        if name_column.startswith("mag_"):
            tab_ap[name_column] = (u.ABmag*column).to(u.nJy).value
            unit = u.nJy
            rename[name_column] = f"flux_{name_column[4:]}"
        elif name_column.startswith("magerr_"):
            tab_ap[name_column][~column.mask] = fluxErrFromABMagErr(
                np.array(tab_ap[name_column.replace("err", "")][~column.mask]).astype(float),
                np.array(column[~column.mask]).astype(float),
            )
            unit = u.nJy
            rename[name_column] = f"fluxerr_{name_column[7:]}"
        else:
            unit = None
    else:
        desc, unit = desc
    column.description = desc
    column.unit = unit
tab_ap.rename_columns(list(rename.keys()), list(rename.values()))

coords = [
    SpherePoint(ra, dec, degrees) for ra, dec in zip(tab_ap["RA"], tab_ap["Dec"])
]
tracts = np.array([skymapInfo.findTract(coord).tract_id for coord in coords])
tab_ap["tract"] = tracts
patches = np.empty_like(tracts)

for tract in np.unique(tracts):
    tractInfo = skymapInfo[tract]
    within = tracts == tract
    tab_sub = tab_ap[within]
    coords_sub = list(compress(coords, within))
    tab_sub["patch"] = np.array([
        tractInfo.findPatch(coord).getSequentialIndex() for coord in coords_sub
    ])
    tab_sub["patch"].description = f"{skymap} patch index"

    tab_arrow = astropy_to_arrow(tab_sub)
    row_group_size = compute_row_group_size(tab_arrow.schema)

    pq.write_table(tab_arrow, f"{name_tab}_{skymap}_{tract}.parq", row_group_size=row_group_size)
