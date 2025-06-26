import astropy.table as apTab
import astropy.units as u
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
from lsst.geom import degrees, SpherePoint
import numpy as np
import pyarrow.parquet as pq

skymap = "lsst_cells_v1"
tract = 5063
name_tab = "hlsp_hlf_hst_60mas_goodss_v2.1_catalog"
butler = dafButler.Butler("/repo/main", collections="skymaps")
tractInfo = butler.get("skyMap", skymap=skymap)[tract]

tab_ap = apTab.Table.read(f"{name_tab}.fits")

columns = {
    "id": ("Unique identifier", ""),
    "x": ("X centroid in image coordinates", "pix"),
    "y": ("Y centroid in image coordinates", "pix"),
    "ra": ("R.A. J2000", "deg"),
    "dec": ("Decl. J2000", "deg"),
    "ra_gaia": ("R.A. J2000, corrected by Gaia astrometry following ra_gaia(deg) = R.A.(deg) +0.1130/3600", "deg"),
    "dec_gaia": ("Decl. J2000, corrected by Gaia astrometry following dec_gaia(deg) = decl.(deg) -0.26/3600", "deg"),
    "faper_f160w": ("F160W flux within a 0.7\" aperture", "nJy"),
    "eaper_f160w": ("1sigma F160Werror within a 0.7\" aperture", "nJy"),
    "faper_f850lp": ("F850LP flux within a 0.7\" aperture", "nJy"),
    "eaper_f850lp": ("1sigma F850LP error within a 0.7\" aperture", "nJy"),
    "tot_cor": ("Inverse fraction of light enclosed at the circularized Kron radius", ""),
    "wmin_hst": ("Minimum weight for ACS and WFC3 bands (excluding zero exposure)", ""),
    "nfilt_hst": ("Number of HST filters with non-zero weight", ""),
    "z_spec": ("Spectroscopic redshift, when available (details in Skelton et al. 2014)", ""),
    "star_flag": ("Point source = 1, extended source = 0 for objects with total H_F160W <or= 25 mag (All objects with H_F160W > 25 mag or no F160W/F850LP coverage have star_flag = 2)", ""),
    "kron_radius": ("SExtractor KRON_RADIUS (pixels)", "pix"),
    "a_image": ("Semimajor axis (SExtractor A_IMAGE)", "pix"),
    "b_image": ("Semiminor axis (SExtractor B_IMAGE)", "pix"),
    "theta_J2000": ("Position angle of the major axis (counter-clockwise, measured from East)", ""),
    "class_star": ("Stellarity index (SExtractor CLASS_STAR parameter)", ""),
    "flux_radius": ("Circular aperture radius enclosing half the total flux (SExtractor FLUX_RADIUS parameter)", "pix"),
    "fwhm_image": ("FWHM from a Gaussian fit to the core (SExtractor FWHM parameter)", "pix"),
    "flags": ("SExtractor extraction flags (SExtractor FLAGS parameter)", ""),
    "detection_flag": ("A flag indicating whether the corrections and structural parameters were derived from F850LP rather than F160W (1 = F850LP, 0 = F160W)", ""),
    "use_f160w": ("Flag indicating source is likely to be a galaxy with reliable measurements in >or=5 filters with (S/N)_F160W > 3 (see text)", ""),
    "use_f850lp": ("Flag indicating source is detected with (S/N)_F850LP > 3 (in at least 1 filter) and likely to be a galaxy (see text)", ""),
}

columns_flux = {
    "f_{band}": ("Total flux in the {band} band", "nJy"),
    "e_{band}": ("1sigma flux error in the {band} band", "nJy"),
    "w_{band}": ("Weight relative to 95th percentile exposure within {band}-band image (see text)", ""),
}

bands = [column.split("f_", 1)[1] for column in tab_ap.columns if column.startswith("f_")]

flux_factor = (25*u.ABmag).to(u.nJy).value

for band in bands:
    for column, (desc, unit) in columns_flux.items():
        columns[column.format(band=band)] = (desc.format(band=band), unit)

for name_column, (desc, unit) in columns.items():
    column = tab_ap.columns[name_column]
    column.unit = unit
    if unit == "nJy":
        bad = column.data == -99
        column.data[bad] = np.nan
        column.data[~bad] *= flux_factor
    column.description = desc

for name_column in ("detection_flag", "use_f160w", "use_f850lp"):
    column = tab_ap[name_column]
    tab_ap[name_column] = tab_ap[name_column].data.astype(bool)
    tab_ap[name_column].unit = column.unit
    tab_ap[name_column].description = column.description

coords = [
    SpherePoint(ra, dec, degrees) for ra, dec in zip(tab_ap["ra_gaia"], tab_ap["dec_gaia"])
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

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, f"{name_tab}.parq", row_group_size=row_group_size)
