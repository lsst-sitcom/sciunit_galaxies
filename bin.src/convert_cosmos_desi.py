import astropy.table as apTab
import astropy.units as u
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
from lsst.geom import degrees, SpherePoint
import numpy as np
import pyarrow.parquet as pq

filename = "DESI-COSMOS-v2.0"
path = f"{filename}.fits"

skymap = "lsst_cells_v2"
tract = 9813
butler = dafButler.Butler("main", collections="skymaps")
tractInfo = butler.get("skyMap", skymap=skymap)[tract]


unit_substitutes = {}

unit_conversions = {
    u.Unit("1e-17 erg/s/cm**2"): u.nJy,
    u.Unit("1e34 cm**4 s**2/erg**2"): u.nJy**2,
}

dtype_substitutes = {
    "str": "S",
}

columns = {
    1: {
        "TARGETID": ("", "int64", "DESI target ID"),
        "DESINAME": ("", "str","Human readable identifier of a sky location DESI $JXXX.XXXX[+/-]YY.YYYY$, where X,Y=truncated decimal TARGET\_RA, TARGET\_DEC, precise to 0.36 arcsec. Multiple objects can map to a single DESINAME if very close on the sky."),
        "TARGET_RA": ("deg","float64","Barycentric right ascension in ICRS"),
        "TARGET_DEC": ("deg","float64","Barycentric declination in ICRS"),
        "OBJTYPE": ("", "str","Object type: TGT, SKY, NON, BAD"),
        "PMRA": ("mas / yr","float32","Proper motion in the +RA direction (includes cos(dec))"),
        "PMDEC": ("mas / yr","float32","Proper motion in the +Dec direction"),
        "SURVEY": ("", "str","Survey of the best spectrum (Main, SV1, SV3, Special)"),
        "PROGRAM": ("", "str","Program of the best spectrum (bright, dark, backup, other)"),
        "HEALPIX": ("", "int32","HealPIX pixel of the best spectrum"),
        "Z": ("", "float64","Redshift from Redrock or Redrock+afterburners"),
        "ZERR": ("", "float64","Redshift uncertainty"),
        "ZWARN": ("", "int64","Redshift warning flag; 0 = no issue, 4 = low DELTACHI2"),
        "DELTACHI2": ("", "float64","Δχ² between best and second-best PCA template fits"),
        "CHI2": ("", "float64","Non-reduced χ² from best PCA template fit"),
        "NPIXELS": ("", "int64","Number of unmasked pixels used in Redrock fit"),
        "SPECTYPE": ("", "str","Spectral classification from Redrock"),
        "SUBTYPE": ("", "str","Spectral subtype (may be blank)"),
        "BEST_Z": ("", "float64","Best reported redshift"),
        "DZ": ("", "float64","Largest redshift difference from merged objects"),
        "QUALITY_Z": ("", "bool","Flag indicating if redshift is quality (0 or 1)"),
        "TSNR2_BGS": ("", "float32","Target S/N² for Bright Galaxy Survey"),
        "TSNR2_LRG": ("", "float32","Target S/N² for Luminous Red Galaxy"),
        "TSNR2_ELG": ("", "float32","Target S/N² for Emission Line Galaxy"),
        "TSNR2_QSO": ("", "float32","Target S/N² for Quasar"),
        "TSNR2_LYA": ("", "float32","Target S/N² for Lyman-alpha"),
        "IS_QSO_MGII": ("", "bool","Flag for quasar identified by MgII afterburner"),
        "A": ("", "float32","MgII fit parameter A"),
        "B": ("", "float32","MgII fit parameter B"),
        "Z_NEW": ("", "float64","New redshift from Redrock (QN prior, QSO templates)"),
        "ZERR_NEW": ("", "float32","Error on Z_NEW"),
        "Z_QN": ("", "float32","Redshift from Quasarnp"),
        "IS_QSO_QN_NEW_RR": ("", "bool","QSO from Quasarnp and new Redrock+QN fit"),
        "DESI_TARGET": ("", "int64","DESI target selection bitmask"),
        "BGS_TARGET": ("", "int64","BGS target selection bitmask"),
        "MWS_TARGET": ("", "int64","Milky Way Survey targeting bits"),
        "SCND_TARGET": ("", "int64","Secondary programs target selection bitmask"),
        "SV1_DESI_TARGET": ("", "int64","DESI SV1 target selection"),
        "SV1_BGS_TARGET": ("", "int64","BGS SV1 target selection"),
        "SV1_MWS_TARGET": ("", "int64","MWS SV1 targeting bits"),
        "SV1_SCND_TARGET": ("", "int64","Secondary SV1 target selection"),
        "SV3_DESI_TARGET": ("", "int64","DESI SV3 target selection"),
        "SV3_BGS_TARGET": ("", "int64","BGS SV3 target selection"),
        "SV3_MWS_TARGET": ("", "int64","MWS SV3 targeting bits"),
        "SV3_SCND_TARGET": ("", "int64","Secondary SV3 target selection"),
        "SPECIAL_TARGET": ("", "str","Target selection description for special programs"),
        "OII_3726_FLUX": ("1e-17 erg/s/cm**2","float32","OII_3726 flux from FastSpecFit"),
        "OII_3726_FLUX_IVAR": ("1e34 cm**4 s**2/erg**2", "float32","Inverse variance of flux"),
        "OII_3729_FLUX": ("1e-17 erg/s/cm**2","float32","OII_3729 flux from FastSpecFit"),
        "OII_3729_FLUX_IVAR": ("1e34 cm**4 s**2/erg**2", "float32","Inverse variance of flux"),
        "OIII_4959_FLUX": ("1e-17 erg/s/cm**2","float32","OIII_4959 flux from FastSpecFit"),
        "OIII_4959_FLUX_IVAR": ("1e34 cm**4 s**2/erg**2", "float32","Inverse variance of flux"),
        "OIII_5007_FLUX": ("1e-17 erg/s/cm**2","float32","OIII_5007 flux from FastSpecFit"),
        "OIII_5007_FLUX_IVAR": ("1e34 cm**4 s**2/erg**2", "float32","Inverse variance of flux"),
        "HALPHA_FLUX": ("1e-17 erg/s/cm**2","float32","HALPHA flux from FastSpecFit"),
        "HALPHA_FLUX_IVAR": ("1e34 cm**4 s**2/erg**2", "float32","Inverse variance of flux"),
        "HBETA_FLUX": ("1e-17 erg/s/cm**2","float32","HBETA flux from FastSpecFit"),
        "HBETA_FLUX_IVAR": ("1e34 cm**4 s**2/erg**2", "float32","Inverse variance of flux"),
        "TOTAL_NUM_COADD": ("", "float64","Number of coadded exposures"),
        "TOTAL_COADD_EXPTIME": ("s","float64","Total exposure time (s)"),
        "LRG_MASK": ("", "uint8","Bright star mask for LRG targets"),
        "ELG_MASK": ("", "uint8","Bright star mask for ELG targets"),
        "HAS_RVS": ("", "bool","Indicates if RVS measurements exist"),
        "VRAD": ("km/s","float64","Radial velocity"),
        "VRAD_ERR": ("km/s","float64","Radial velocity error"),
        "VRAD_SKEW": ("", "float64","Radial velocity skewness"),
        "VRAD_KURT": ("", "float64","Radial velocity kurtosis"),
        "LOGG": ("", "float64","log(surface gravity)"),
        "TEFF": ("K","float64","Effective temperature"),
        "FE_H": ("", "float64","[Fe/H] metallicity"),
        "LOGG_ERR": ("", "float64","Error on LOGG"),
        "TEFF_ERR": ("K","float64","Error on TEFF"),
        "FE_H_ERR": ("", "float64","Error on FE_H"),
        "VSINI": ("km/s","float64","Stellar rotational velocity"),
        "RVS_WARN": ("", "int64","RVSpecfit warning flag"),
        "HAS_DECALS_DR9": ("", "bool","Has DECaLS DR9 photometry"),
        "HAS_DECAM_DR10": ("", "bool","Has DECam DR10 photometry"),
        "HAS_HSC_UD_PDR3": ("", "bool","Has HSC UltraDeep PDR3 photometry"),
        "HAS_HSC_WIDE_PDR3": ("", "bool","Has HSC Wide PDR3 photometry"),
        "HAS_COSMOS2020": ("", "bool","Has COSMOS2020 photometry"),
        "HAS_MERIAN": ("", "bool","Has Merian photometry"),
        "HAS_VI_Z": ("", "bool","Flag for visually inspected redshift"),
        "VI_Z": ("", "float64","Visually inspected redshift"),
        "VI_SPECTYPE": ("", "str","Visually inspected spectral type"),
        "VI_QUALITY": ("", "float64","Visual redshift quality (3 or 4 = high quality)"),
    },
}

tables = []
n_columns = 0
meta_inputs = {}

for idx_tab, columns_tab in columns.items():
    tab_ap = apTab.Table.read(path, hdu=idx_tab)
    columns_add = {}
    columns_del = set()
    columns_fix = {}

    for idx in range(len(tab_ap.columns)):
        column = tab_ap.columns[idx]
        name = column.name
        unit, dtype, description = columns_tab[name]

        if name.startswith("HAS_"):
            print(f"{name} column will be deleted as it is unnecessary")
            columns_del.add(name)
            continue

        columns_fix[name] = unit, description, dtype

    for column in columns_del:
        del tab_ap[column]
    for column, (values, unit) in columns_add.items():
        tab_ap[column] = values
        if unit is not None:
            tab_ap[column].unit = unit
    for name_fix, (unit, description_new, dtype_schema) in columns_fix.items():
        dtype_schema = dtype_substitutes.get(dtype_schema, dtype_schema)
        column = tab_ap[name_fix]
        column.description = description_new
        column.byteswap(inplace=True)
        column.dtype = column.dtype.newbyteorder()
        dtype = column.dtype

        if (unit_new := unit_conversions.get(unit)) is not None:
            try:
                factor = unit.to(unit_new, 1.0)
                tab_ap[name] *= factor
                print(f"multiplying {name} values by {factor=}")
                unit = unit_new
            except Exception as exc:
                print(f"converting unit_new={unit} got {exc=}")
        column.unit = unit

        if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.unsignedinteger):
            if (np.nanmin(column.data) == 0) and (np.nanmax(column.data) == 1) and (
                    dtype != bool): # noqa E721
                print(f"{column.name} column changing from {dtype=} to bool")
                tab_ap[name_fix] = column.astype(bool)

        if not np.issubdtype(dtype, dtype_schema):
            raise RuntimeError(f"{column.name} {dtype=} not expected {dtype_schema=}")

    meta_inputs[idx_tab] = {
        "columns": f"{n_columns}:{n_columns + len(tab_ap.colnames)}",
        "EXTNAME": tab_ap.meta["EXTNAME"],
    }
    tab_ap.meta = {}
    tables.append(tab_ap)
    n_columns += len(tab_ap.colnames)

if len(tables) > 1:
    tab_ap = apTab.hstack(tables)
    tab_ap.meta["inputs"] = meta_inputs


for column in ["TARGET_RA", "TARGET_DEC"]:
    column_error = f"{column}_est_error"
    tab_ap[column_error] = np.full(len(tab_ap), 0.01 / 3600, dtype=np.float32)
    tab_ap[column_error].description = f"Placeholder {column_error} error (constant 10 mas)"
    tab_ap[column_error].unit = u.deg

coords = [
    SpherePoint(ra, dec, degrees) for ra, dec in zip(tab_ap["TARGET_RA"], tab_ap["TARGET_DEC"])
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

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, f"{filename}.parq", row_group_size=row_group_size)
