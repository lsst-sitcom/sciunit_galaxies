import astropy.table
import astropy.units as u
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, astropy_to_arrow, pa, pq
import numpy as np

truth_summary_path = "/sdf/data/rubin/shared/dc2_run2.2i_truth/truth_summary_cell"

plot = False

bands = ("u", "g", "r", "i", "z", "y")
mag_total_min_star = 17.5
mag_total_min_galaxy = 15
mag_total_max = 26.5

tract_in = 3828
tract_out = 9813

skymap_name_in = "DC2_cells_v1"
skymap_name_out = "hsc_rings_v1"

butler_in = dafButler.Butler("/repo/dc2")
butler_out = dafButler.Butler("/repo/main", collections=["HSC/runs/RC2/w_2024_38/DM-46429"])

skymap_in, skymap_out = (
    butler.get("skyMap", skymap=skymap_name, collections="skymaps")
    for butler, skymap_name in ((butler_in, skymap_name_in), (butler_out, skymap_name_out))
)

tractinfo_in, tractinfo_out = (
    skymap[tract] for skymap, tract in ((skymap_in, tract_in), (skymap_out, tract_out))
)

(cen_ra_in, cen_dec_in), (cen_ra_out, cen_dec_out) = (
    (x.asDegrees() for x in tractinfo.ctr_coord) for tractinfo in (tractinfo_in, tractinfo_out)
)

truth_summary = arrow_to_astropy(pq.read_table(
    f"{truth_summary_path}/truth_summary_v2_{tract_in}_{skymap_name_in}_2_2i_truth_summary.parq"
))

# Dump some unneeded columns
for column in ((
    "id_string", "host_galaxy", "redshift", "A_V", "R_V",
    "tract", "patch", "cosmodc2_hp", "cosmodc2_id",
    "ra_unlensed", "dec_unlensed", "redshift_Hubble",
)):
    if column in truth_summary.colnames:
        del truth_summary[column]

# Stars and galaxies only; no SN
truth_galaxy = truth_summary["truth_type"] == 1
truth_star = truth_summary["truth_type"] == 2
truth_good = truth_star | truth_galaxy
truth_summary = truth_summary[truth_good]

# Cut out very faint objects
flux_total = np.sum([truth_summary[f"flux_{band}"] for band in "ugrizy"], axis=0)
mag_total = u.nJy.to(u.ABmag, flux_total)
truth_out = truth_summary[
    (mag_total > (mag_total_min_star*truth_star[truth_good] + mag_total_min_galaxy*truth_galaxy[truth_good]))
    & (mag_total < mag_total_max)
]

ra_in, dec_in = (truth_out[col] for col in ("ra", "dec"))
dec_out = dec_in + cen_dec_out - cen_dec_in
ra_out = cen_ra_out + (ra_in - cen_ra_in)*np.cos(dec_in*np.pi/180)/np.cos(dec_out*np.pi/180)

if plot:
    import matplotlib.pyplot as plt
    objects = butler_out.get(
        "objectTable_tract", skymap=skymap_name_out, tract=tract_out,
        parameters={"columns": ["coord_ra", "coord_dec"]}, storageClass="ArrowAstropy",
    )
    plt.scatter(ra_out[::20], dec_out[::20], s=1.5)
    plt.scatter(objects["coord_ra"][::20], objects["coord_dec"][::20], s=1.5)
    plt.show()

is_star = truth_out["truth_type"] == 2
is_galaxy = truth_out["truth_type"] == 1

n_star = np.sum(is_star)
n_galaxy = np.sum(is_galaxy)


def concatenate_star_galaxy(values, is_star, is_galaxy):
    values_galaxy = values[is_galaxy]
    return np.concatenate((values[is_star], values_galaxy, values_galaxy))


source_types = ["DeltaFunction"]*n_star
source_types.extend(["Sersic"]*(2*n_galaxy))

mask_star = np.concatenate((np.ones(n_star, dtype=bool), np.zeros(2*n_galaxy, dtype=bool)))

# GalSim convention is x-axis rather than y-axis
positionAngles = 90 + truth_out["positionAngle"][is_galaxy]
positionAngles[positionAngles < 0] += 180

data_out = {
    "injection_id": np.arange(2*n_galaxy + n_star),
    "group_id": concatenate_star_galaxy(truth_out["id"], is_star, is_galaxy),
    "ra": concatenate_star_galaxy(ra_out, is_star, is_galaxy),
    "dec": concatenate_star_galaxy(dec_out, is_star, is_galaxy),
    "source_type": source_types,
    "n": np.ma.masked_array(
        np.concatenate((
            np.zeros(n_star, dtype=float),
            np.ones(n_galaxy, dtype=float),
            np.full(n_galaxy, 4.0, dtype=float),
        )),
        mask=mask_star,
        fill_value=np.nan,
    ),
    "half_light_radius": np.ma.masked_array(
        np.concatenate((
            np.zeros(n_star, dtype=float),
            truth_out["diskMajorAxisArcsec"][is_galaxy]*np.sqrt(truth_out["diskAxisRatio"][is_galaxy]),
            truth_out["spheroidMajorAxisArcsec"][is_galaxy]*np.sqrt(
                truth_out["spheroidAxisRatio"][is_galaxy]
            ),
        )),
        mask=mask_star,
        fill_value=0.,
    ),
    "q": np.ma.masked_array(
        np.concatenate((
            np.ones(n_star, dtype=float),
            truth_out["diskAxisRatio"][is_galaxy],
            truth_out["spheroidAxisRatio"][is_galaxy],
        )),
        mask=mask_star,
        fill_value=1.,
    ),
    "beta": np.ma.masked_array(
        np.concatenate((
            np.ones(n_star, dtype=float),
            positionAngles,
            positionAngles,
        )),
        mask=mask_star,
        fill_value=0.,
    ),
}

table_out = astropy.table.Table(data_out)

for column, (description, unit) in {
    "injection_id": ("Injection object ID (row number)", None),
    "group_id": ("Injection object ID (row number)", None),
    "ra": ("Right ascension", "deg"),
    "dec": ("Declination", "deg"),
    "source_type": ("Injection source type", None),
    "n": ("Sersic index", None),
    "half_light_radius": ("Sersic half-light radius [sqrt(a*b)]", u.arcsec),
    "q": ("Minor-to-major axis ratio", None),
    "beta": ("Position angle", u.deg),
}.items():
    column = table_out[column]
    column.description = description
    if unit is not None:
        column.unit = unit

for band in bands:
    bulgefrac = truth_out[f"bulge_to_total_{band}"].data.data[is_galaxy]
    fluxes = truth_out[f"flux_{band}"]
    table_out["mag"] = np.concatenate((
        fluxes[is_star].to(u.ABmag),
        (fluxes[is_galaxy]*(1-bulgefrac)).to(u.ABmag),
        (fluxes[is_galaxy]*bulgefrac).to(u.ABmag),
    ))
    table_out["mag"].description = f"{band}-band magnitude"
    table_out["mag"].unit = u.ABmag
    filename = f"injection_catalog_{skymap_name_out}_{tract_out}_from_{skymap_name_in}_{band}_{tract_in}.parq"
    print(f"Writing {filename}")
    pq.write_table(astropy_to_arrow(table_out), filename)
