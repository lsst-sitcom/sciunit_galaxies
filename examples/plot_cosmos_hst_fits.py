import glob
import os

from astropy.coordinates import SkyCoord
from astropy.table import join, Table
from lsst.afw.table import SourceCatalog
from lsst.daf.butler.formatters.parquet import arrow_to_astropy
from lsst.gauss2d.utils import covar_to_ellipse
from lsst.sitcom.sciunit.galaxies.read_cosmos_data import get_dataset_filepath
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


testdata_cosmos_dir = os.environ["TESTDATA_COSMOS_DIR"]
tract: int = 9813
patch: int = 40
kwargs_path = {"tract": tract, "patch": patch}

suffix = "*.fits"
filepath = get_dataset_filepath(dataset="deepCoadd_ref", suffix=suffix, **kwargs_path)
catalog_ref_hsc = SourceCatalog.readFits(
    next(iter(glob.glob(f"{testdata_cosmos_dir}/{filepath}")))
).asAstropy()
ra_hsc, dec_hsc = (catalog_ref_hsc[col] for col in ("coord_ra", "coord_dec"))
radec_min = SkyCoord(np.min(ra_hsc)*180/np.pi, np.min(dec_hsc)*180/np.pi, unit="deg")
radec_max = SkyCoord(np.max(ra_hsc)*180/np.pi, np.max(dec_hsc)*180/np.pi, unit="deg")

tile_table = Table.read(f"{testdata_cosmos_dir}/cosmos_hst_tiles.ecsv")
path_tiles = f"{testdata_cosmos_dir}/cosmos_hst_tiles/"
tiles_within = [
    (f'{path_tiles}{row["path_science"]}', f'{path_tiles}{row["path_weight"]}') for row in tile_table[
        ((tile_table["ra_min"] < radec_max.ra.value) & (tile_table["ra_max"] > radec_min.ra.value))
        & ((tile_table["dec_min"] < radec_max.dec.value) & (tile_table["dec_max"] > radec_min.dec.value))
    ]
]

catalog_hst = arrow_to_astropy(pq.read_table(f"{testdata_cosmos_dir}/cosmos_acs_iphot_200709.parq"))
within_catalog = (catalog_hst["ra"] > radec_min.ra) & (catalog_hst["ra"] < radec_max.ra) & (
    catalog_hst["dec"] > radec_min.dec) & (catalog_hst["dec"] < radec_max.dec)
catalog_hst = catalog_hst[within_catalog]

filepath_psf = get_dataset_filepath(dataset="cosmos_hst_psf_models_multiprofit", **kwargs_path)
results = arrow_to_astropy(pq.read_table(f"{testdata_cosmos_dir}/{filepath_psf}"))
found = results[results["row_catalog_hst"] >= 0]
catalog_hst["row"] = np.arange(len(catalog_hst))
joined = join(
    found, catalog_hst, join_type="left", keys=None, keys_left=["row_catalog_hst"], keys_right=["row"],
)
joined = join(joined, catalog_ref_hsc[["id", "coord_ra", "coord_dec"]], join_type="left", keys="id")
for column in ("coord_ra", "coord_dec"):
    joined[column] *= 180/np.pi


def make_plots(joined):
    deg2mas = 3600000

    fig, ax = plt.subplots(ncols=1)
    ax.scatter(
        (joined["mpf_psf_cen_x"] - joined["ra"]) * deg2mas,
        (joined["mpf_psf_cen_y"] - joined["dec"]) * deg2mas,
    )
    ax.set_xlabel("mpf RA - HST catalog RA (mas)")
    ax.set_ylabel("mpf Dec - HST catalog Dec (mas)")

    fig1, ax1 = plt.subplots(ncols=2)
    fig2, ax2 = plt.subplots(ncols=2)
    fig_astrom_ref, ax_astrom_ref = plt.subplots(ncols=2)

    coords_hsc = ("ra", "dec")
    coords_mpf = ("x", "y")

    comps_mpf = np.arange(1, 4)

    for idx_coord, (axis1, axis2, axis_astrom_ref) in enumerate(zip(ax1, ax2, ax_astrom_ref)):
        cen = coords_mpf[idx_coord]
        coord = coords_hsc[idx_coord]
        coord_other = coords_hsc[1 - idx_coord]

        diff = (joined[f"mpf_psf_cen_{cen}"] - joined[f"coord_{coord}"]) * deg2mas
        axis1.scatter(joined[f"coord_{coord}"], diff, c=joined[f"coord_{coord_other}"])
        axis1.set_title(f"delta {coord}, color-coded by {coord_other}")
        axis1.set_xlabel(f"HSC {coord} (deg)")
        axis1.set_ylabel(f"mpf {coord} - HSC catalog {coord} (mas)")

        axis2.scatter(joined[f"coord_ra"], joined[f"coord_dec"], c=diff)
        axis2.set_title(f"delta {coord} (asec, MPF HST - HSC)")
        axis2.set_xlabel("ra (HSC)")
        axis2.set_ylabel("dec (HSC)")

        diff_ref = (joined[f"{coord}"] - joined[f"coord_{coord}"]) * deg2mas

        axis_astrom_ref.scatter(joined[f"coord_ra"], joined[f"coord_dec"], c=diff_ref)
        axis_astrom_ref.set_title(f"delta {coord} (asec, HST ref - HSC)")
        axis_astrom_ref.set_xlabel("ra (HSC)")
        axis_astrom_ref.set_ylabel("dec (HSC)")

    fig3, ax3 = plt.subplots(ncols=1)
    flux_mpf = sum(joined[f"mpf_psf_{idx}_F814W_flux"] for idx in comps_mpf)
    mag_mpf = -2.5 * np.log10(flux_mpf) + 25.9
    ax3.scatter(joined["mag_best"], (mag_mpf - joined["mag_best"])*1000)
    ax3.set_xlabel("mag_best HST")
    ax3.set_xlabel("MPF mag - ref (mmag)")
    ax3.set_title("MPF versus ref mags")

    n_comps_mpf = len(comps_mpf)

    sigma_max = deg2mas*np.max([
        np.sqrt(sum(
            joined[f"mpf_psf_{idx}_sigma_{coord}"]**2 for coord in ("x", "y")
        ))
        for idx in comps_mpf
    ])

    fig4, ax4 = plt.subplots(ncols=n_comps_mpf)
    fig5, ax5 = plt.subplots(ncols=n_comps_mpf)

    for idx_comp, axis, axis2 in zip(comps_mpf, ax4, ax5):
        prefix = f"mpf_psf_{idx_comp}_"
        frac_flux = joined[f"{prefix}F814W_flux"]/flux_mpf
        sigma_x, sigma_y = (joined[f"{prefix}sigma_{coord}"]*deg2mas for coord in ("x", "y"))
        rho = joined[f"{prefix}rho"]
        sigma = np.sqrt(sigma_x**2 + sigma_y**2)
        r_major, axrat, angle = covar_to_ellipse(sigma_x_sq=sigma_x**2, sigma_y_sq=sigma_y**2,
                                                 cov_xy=sigma_x*sigma_y*rho)

        axis.scatter(frac_flux, sigma, c=axrat)
        axis.set_xlabel("flux frac")
        axis.set_ylabel("sigma (mas)")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, sigma_max)

        axis2.scatter(sigma_x, sigma_y, c=frac_flux)
        axis2.set_xlabel("sigma_x (mas)")
        axis2.set_ylabel("sigma_y (mas)")
        axis2.set_xlim(0, sigma_max)
        axis2.set_ylim(0, sigma_max)

    plt.show()

make_plots(joined)

pass
