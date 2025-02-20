import astropy.table
import astropy.units as u
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, pq
import lsst.geom
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable


def convert_truth_summary_v2_to_injection(
    bands: Iterable[str] | None = None,
    butler_in: dafButler.Butler | None = None,
    butler_out: dafButler.Butler | None = None,
    skymap_name_in: str = "DC2_cells_v1",
    skymap_name_out: str = "hsc_rings_v1",
    tract_in: float = 3828,
    tract_out: float = 9813,
    mag_total_min_star: float = 17.5,
    mag_total_min_galaxy: float = 15,
    mag_total_max: float = 26.5,
    mag_total_max_component: float = 29.,
    truth_summary_path: str = "/sdf/data/rubin/shared/dc2_run2.2i_truth/truth_summary_cell",
    plot: bool = False,
) -> dict[str, astropy.table.Table]:
    """ Convert a truth_summary_v2 to an injection catalog.

    Parameters
    ----------
    bands
        The bands to create injection catalogs for.
    butler_in
        The butler containing the skymap definition for the truth_summary_v2.
    butler_out
        The butler containing the skymap definition for the injection catalogs.
    skymap_name_in
        The name of the skymap for the input truth_summary_v2.
    skymap_name_out
        The name of the skymap for the output injection catalogs.
    tract_in
        The input tract number.
    tract_out
        The output tract number.
    mag_total_min_star
        The minimum total magnitude for stars to be included (i.e. bright cutoff).
    mag_total_min_galaxy
        The minimum total magnitude for galaxies to be included (i.e. bright cutoff).
    mag_total_max
        The maximum total magnitude for any object to be included (i.e. faint cutoff).
    mag_total_max_component
        The maximum magnitude for any single component to be included (i.e. faint cutoff).
    truth_summary_path
        The path to parquet summary files (if not butler ingested).
    plot
        Whether to make a plot of the object overlap in the tract.

    Returns
    -------
    catalogs
        The injection catalogs, keyed by filename, in the order of bands.
    mags
        If the output table differs only in the magnitude column, this dict
        will contain magnitude values by band; otherwise it will be empty.
    """
    if bands is None:
        bands = ("u", "g", "r", "i", "z", "y")
    if butler_in is None:
        butler_in = dafButler.Butler("/repo/dc2")
    if butler_out is None:
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
        (mag_total > (mag_total_min_star*truth_star[truth_good] +
                      mag_total_min_galaxy*truth_galaxy[truth_good]))
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
    n_rows = 2*n_galaxy + n_star

    data_out = {
        "injection_id": np.arange(n_rows),
        "group_id": concatenate_star_galaxy(truth_out["id"], is_star, is_galaxy),
        "ra": concatenate_star_galaxy(ra_out, is_star, is_galaxy),
        "dec": concatenate_star_galaxy(dec_out, is_star, is_galaxy),
        "mag": np.zeros(n_rows, dtype=float),
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
        "mag": ("magnitude", u.ABmag),
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

    tables_out = {}
    mags = {}
    prefix = "injection_catalog_"
    for band in bands:
        bulgefrac = truth_out[f"bulge_to_total_{band}"].data.data[is_galaxy]
        fluxes = truth_out[f"flux_{band}"]
        mags[band] = np.concatenate((
            fluxes[is_star].to(u.ABmag),
            (fluxes[is_galaxy]*(1-bulgefrac)).to(u.ABmag),
            (fluxes[is_galaxy]*bulgefrac).to(u.ABmag),
        ))
        filename = f"{prefix}{skymap_name_out}_{tract_out}_from_{skymap_name_in}_{band}_{tract_in}.parq"
        tables_out[filename] = table_out

    if mag_total_max_component < np.Inf:
        mag_total = u.nJy.to(
            u.ABmag,
            np.sum([mags_band.unit.to(u.nJy, mags_band.value) for mags_band in mags.values()], axis=0)
        )
        good = mag_total < mag_total_max_component
        table_out = table_out[good]
        for (band, mags_band), filename in zip(mags.items(), tables_out.keys()):
            mags[band] = mags_band[good]
            tables_out[filename] = table_out

    return tables_out, mags


def validate_injection_catalog(
    band: str = "r",
    butler_in: dafButler.Butler | None = None,
    butler_out: dafButler.Butler | None = None,
    skymap_name_in: str = "DC2_cells_v1",
    skymap_name_out: str = "hsc_rings_v1",
    tract_in: int = 3828,
    tract_out: int = 9813,
    patch: int = 40,
    cutout_asec: float = 60,
    ids_ref: Iterable[int] | None = None,
    truth_summary_path: str = "/sdf/data/rubin/shared/dc2_run2.2i_truth/truth_summary_cell",
):
    if butler_in is None:
        butler_in = dafButler.Butler(
            "/repo/dc2", skymap=skymap_name_in, collections=["2.2i/runs/test-med-1/w_2024_40/DM-46604"],
        )
    if butler_out is None:
        butler_out = dafButler.Butler("/repo/main", skymap=skymap_name_out,
                                      collections=["u/dtaranu/DM-44943/injected"])
    if ids_ref is None:
        ids_ref = [
            7813120317, 7812500182, 7812541653, 7812608827, 7812509574,
            7812509636, 7812612080, 7812647026, 7812647658, 7813026314,
        ]
    injection = arrow_to_astropy(
        pq.read_table(
            f"injection_catalog_{skymap_name_out}_{tract_out}_from_{skymap_name_in}_{band}_{tract_in}.parq"
        )
    )
    calexp_out = butler_out.get("deepCoadd_calexp", patch=patch, tract=tract_out, band=band)
    calexp_inj = butler_out.get("injected_deepCoadd", patch=patch, tract=tract_out, band=band)
    truth_summary_v2 = arrow_to_astropy(pq.read_table(
        f"{truth_summary_path}/truth_summary_v2_{tract_in}_{skymap_name_in}_2_2i_truth_summary.parq"
    ))

    objects = butler_out.get(
        "objectTable_tract", tract=tract_out, storageClass="ArrowAstropy",
        parameters={"columns": ("coord_ra", "coord_dec", "detect_isPrimary",)},
    )

    plt.scatter(objects["coord_ra"][::20], objects["coord_dec"][::20], s=0.8)
    plt.scatter(injection["ra"][::20], injection["dec"][::20], s=0.8)
    plt.tight_layout()
    plt.show()

    pa_offset = 90
    scale_out = calexp_out.getWcs().getPixelScale().asArcseconds()
    extent_out = lsst.geom.Extent2I(int(cutout_asec/scale_out), int(cutout_asec/scale_out))

    for id_ref in ids_ref:
        obj_ref = injection[injection["group_id"] == id_ref][0]
        ra_out, dec_out = (obj_ref[c] for c in ("ra", "dec"))

        obj_dc2 = truth_summary_v2[truth_summary_v2["id"] == id_ref][0]
        calexp_dc2 = butler_in.get(
            "deepCoadd_calexp", patch=obj_dc2["patch"], tract=obj_dc2["tract"], band=band,
        )
        scale_in = calexp_dc2.getWcs().getPixelScale().asArcseconds()
        extent_in = lsst.geom.Extent2I(int(cutout_asec / scale_in), int(cutout_asec / scale_in))

        cutout_out, cutout_out_inj = (
            calexp.getCutout(
                center=lsst.geom.SpherePoint(ra_out, dec_out, lsst.geom.degrees), size=extent_out,
            )
            for calexp in (calexp_out, calexp_inj)
        )

        coord_dc2 = lsst.geom.SpherePoint(obj_dc2["ra"], obj_dc2["dec"], lsst.geom.degrees)

        cutout_dc2 = calexp_dc2.getCutout(center=coord_dc2, size=extent_in)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

        min_in = np.nanmin(cutout_out.image.array)
        max_in = np.nanmax(cutout_out.image.array)

        pa = obj_dc2['positionAngle']

        ax[0][0].imshow(np.arcsinh(cutout_out.image.array * 10), cmap="gray")
        ax[0][0].set_title(f"{skymap_name_out} {tract_out=} {patch=} {band=}")
        ax[0][1].imshow(np.arcsinh(np.clip(cutout_out_inj.image.array, min_in, max_in) * 10), cmap="gray")
        ax[0][1].set_title(f"{skymap_name_out} injected {tract_out=} {patch=} {band=}")
        ax[1][0].imshow(
            np.arcsinh(10 * np.clip(cutout_out_inj.image.array - cutout_out.image.array, min_in, max_in)),
            cmap="gray",
        )
        ax[1][0].set_title(f"injected difference")
        ax[1][1].imshow(np.arcsinh(np.clip(cutout_dc2.image.array, min_in, max_in) * 10), cmap="gray")
        ax[1][1].set_title(f"DC2 id={id_ref} tract={obj_dc2['tract']} patch={obj_dc2['patch']}"
                           f" {pa_offset}-PA={pa_offset - pa:.1f}")
        coord = [x / 2 for x in cutout_dc2.image.array.shape[::-1]]
        linelen = coord[0] / 8
        dx = np.cos((-pa + pa_offset) * np.pi / 180)
        dy = np.sin((-pa + pa_offset) * np.pi / 180)
        ax[1][1].plot([coord[0] - linelen * dx, coord[0] + linelen * dx],
                      [coord[1] - linelen * dy, coord[1] + linelen * dy], 'r-')

        plt.tight_layout()

    plt.show()
