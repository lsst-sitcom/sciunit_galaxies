from datetime import datetime

from astropy.table import Table, join
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, astropy_to_arrow, pa, pq
from lsst.geom import SpherePoint, degrees
import GCRCatalogs
# from GCRCatalogs.helpers.tract_catalogs import tract_filter
import numpy as np

## conda-installed GCRCatalog v1.8 not compatible with astropy>=6.1
## (fix currently exist in the main branch of GCRCatalogs but not conda-installable as of yet Aug 23, 2024)
## To use the conda-installable version of GCRCatalogs, use astropy<6.1
## import astropy
# astropy.__version__

do_mags = False

butler = dafButler.Butler("/repo/dc2")
name_skymap = "DC2_cells_v1"
skymap = butler.get("skyMap", skymap=name_skymap, collections="skymaps")

GCRCatalogs.set_root_dir('/sdf/data/rubin/user/combet')
print(f"root dir={GCRCatalogs.get_root_dir()}")

truth = GCRCatalogs.load_catalog('desc_dc2_run2.2i_dr6_truth')
tracts = truth.available_tracts
print(f"Available tracts: {tracts}")

truth_quantities = truth.list_all_quantities(include_native=True)
truth_columninfo = {tq: truth.get_quantity_info(tq) for tq in truth_quantities}
truth_columninfo['av'] = {'unit': 'mag'}
truth_columninfo['rv'] = {'unit': 'mag'}

cosmodc2_cat = GCRCatalogs.load_catalog("desc_cosmodc2")
cosmodc2_quantities = cosmodc2_cat.list_all_quantities(include_native=True)

bands = ("u", "g", "r", "i", "z", "y")
components = ("disk", "spheroid")
quantities_morph = ("MajorAxisArcsec", "AxisRatio")

# List of fields from cosmodc2 to add to the default truth table
diskmorph_keys, spheroidmorph_keys = (
    {f"morphology/{comp}{quant}": f"{comp}{quant}" for quant in quantities_morph} for comp in components
)
disklum_keys, spheroidlum_keys = (
    {
        f"LSST_filters/{comp}LuminositiesStellar:LSST_{band}:observed":
            f"{comp}StellarLuminosity_{band}{'_' if suffix else ''}{suffix[1:]}"
        for suffix in (":dustAtlas",) for band in bands
    }
    for comp in components
)
mag_keys = {
    f"LSST_filters/magnitude:LSST_{band}:observed{suffix}":
        f"mag_observed_{band}{'_' if suffix else ''}{suffix[1:]}"
    for suffix in ("", ":dustAtlas") for band in bands
} if do_mags else {}
other_keys = {
    'ra_true': 'ra_unlensed',
    'dec_true': 'dec_unlensed',
    'redshiftHubble': 'redshift_Hubble',
}
key_list = (
    {"morphology/positionAngle": "positionAngle"},
    diskmorph_keys, spheroidmorph_keys,
    disklum_keys, spheroidlum_keys,
    mag_keys, other_keys
)

columns_dict = {
    k: v for key_dict in key_list for k, v in key_dict.items()
}

columns_load = ("galaxy_id",) + tuple(columns_dict.keys())

for key in columns_load:
    assert key in cosmodc2_quantities

columns_dict["av"] = "A_V"
columns_dict["rv"] = "R_V"

print(f"The following quantities will be added to the truth table: {columns_load}")

# Updated descriptions from https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/SCHEMA.md
for column, description in {
    "ra": "Right ascension, lensed (as observed)",
    "dec": "Declination, lensed (as observed)",
    "redshift": "Redshift including line-of-sight proper motion Doppler shift",
    "av": "V-band extinction",
    "rv": "Ratio of absolute to selective extinction: A_V/E(B-V)",
}.items():
    truth_columninfo[column]['description'] = description
for band in bands:
    truth_columninfo[f"flux_{band}"] = {
        'description': f'Observed {band}-band flux (including dust extinction)',
        'unit': 'nJy',
    }
truth_columninfo["id_string"] = {"description": "Original string id", "unit": None}
cosmodc2_columninfo = {
    "morphology/positionAngle": {
        "description": "Position angle relative to north (+Dec) towards east (+RA)",
        "unit": "deg",
    },
    "morphology/diskMajorAxisArcsec": {
        "description": "Disk major axis effective radius",
        "unit": "arcsec",
    },
    "morphology/spheroidMajorAxisArcsec": {
        "description": "Spheroid major axis effective radius",
        "unit": "arcsec",
    },
    "morphology/diskAxisRatio": {
        "description": "Disk minor-to-major axis ratio",
        "unit": "arcsec",
    },
    "morphology/spheroidAxisRatio": {
        "description": "Spheroid minor-to-major axis ratio",
        "unit": "arcsec",
    },
    "ra_true": {
        "description": "Right ascension, unlensed (true, not as observed)",
        "unit": truth_columninfo["ra"]["unit"],
    },
    "dec_true": {
        "description": "Declination, unlensed (true, not as observed)",
        "unit": truth_columninfo["dec"]["unit"],
    },
    "redshiftHubble": {
        "description": "Redshift, cosmological (excluding proper motion Doppler shift)",
        "unit": "none",
    }
}

tracts = (3828, 3829)

for tract in tracts:
    print(f"==== Reading default truth catalog for tract={tract} ====")
    # These already have new integer ID columns
    truth = arrow_to_astropy(pq.read_table(
        f"/sdf/group/rubin/ncsa-project/project/shared/DC2/truth_summary_v2/truth_tract{tract}.parquet")
    )
    filt = truth["is_unique_truth_entry"]
    truth = truth[filt]
    if not do_mags:
        del truth["mag_r"]

    for column in reversed((
        'match_objectId', 'match_sep', 'is_good_match', 'is_nearest_neighbor', 'is_unique_truth_entry',
        'patch',
    )):
        del truth[column]

    for column_name, info in truth_columninfo.items():
        if (column := truth.columns.get(column_name)) is not None:
            column.description = info['description']
            column.unit = info['unit']

    # This should be useable as a filter instead of the hand-select healpix
    # ... but I can't get it to work
    # filter_t = tract_filter(tracts=[tract])
    hp_pix = np.unique(truth['cosmodc2_hp'])
    hp_pix = hp_pix[hp_pix >= 0]
    print(f'Tract {tract} corresponds to the following Healpix pixels in cosmoDC2: {hp_pix}')

    #
    print("==== Reading cosmodc2 ====")
    cosmodc2_cat = GCRCatalogs.load_catalog("desc_cosmodc2")

    cosmodc2 = Table(cosmodc2_cat.get_quantities(
        columns_load,
        native_filters=' | '.join(f'(healpix_pixel == {hp})' for hp in hp_pix),
    ))
    # Re-order columns
    cosmodc2 = cosmodc2[columns_load]
    for column in columns_load:
        units = cosmodc2_cat.get_quantity_info(column)['units']
        if units:
            cosmodc2[column].units = units
    # Apply column info overrides
    for column_name, info in cosmodc2_columninfo.items():
        if (column := cosmodc2.columns.get(column_name)) is not None:
            column.description = info['description']
            column.unit = info['unit']

    print("==== Merging tables according to ID ====")
    cosmodc2.rename_column('galaxy_id', 'cosmodc2_id')
    merged_table = join(truth, cosmodc2, keys='cosmodc2_id', join_type='left')

    print("==== Adding descriptions ====")

    merged_table.rename_columns(tuple(columns_dict.keys()), tuple(columns_dict.values()))
    columns_rename = {}
    for band in bands:
        column_spheroid, column_disk = (
            f"{comp}StellarLuminosity_{band}_dustAtlas" for comp in ("spheroid", "disk")
        )
        merged_table[column_spheroid] /= merged_table[column_spheroid] + merged_table[column_disk]
        del merged_table[column_disk]
        columns_rename[column_spheroid] = f"bulge_to_total_{band}"
        merged_table[column_spheroid].unit = None
        merged_table[column_spheroid].description = f"Fraction of {band}-band luminosity in bulge"
    merged_table.rename_columns(tuple(columns_rename.keys()), tuple(columns_rename.values()))

    merged_table.meta["description"] = "\n".join(
        f"truth_summary_v2 table generated on {str(datetime.utcnow())}."
        f"See DM-44943 for details."
        f"All disks and bulges have n_sersic=1 and 4, respectively."
        f"Both disk and bulge components have the same position angle."
        f"Bulge-to-total ratios are derived from dustAtlas luminosities."
        f"These do not exactly match the magnitudes in run 2.2i."
    )

    print("==== Populating patch column ====")
    tractinfo = skymap[tract]
    radecs = [SpherePoint(ra, dec, degrees) for ra, dec in zip(truth["ra"], truth["dec"])]
    patches = [tractinfo.findPatch(radec).sequential_index for radec in radecs]
    merged_table["patch"] = patches
    merged_table["patch"].description = f"The patch number in {skymap=}"

    filename_out = f"truth_summary_v2_{tract}_{name_skymap}_2_2i_truth_summary.parq"
    pq.write_table(astropy_to_arrow(merged_table), filename_out)

    print("==== Done ====")

    print("==== Read in new merged table file and check for units and descriptions ====")

    pa_table = pa.parquet.read_table(filename_out)
    ap_table = arrow_to_astropy(pa_table)
    print(ap_table['dec'].unit, ap_table['dec'].description)
