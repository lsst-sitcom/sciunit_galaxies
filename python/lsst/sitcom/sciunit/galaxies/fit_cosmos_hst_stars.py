import logging
import math
from typing import Any, ClassVar, Mapping, Sequence

from astropy.coordinates import SkyCoord
from astropy.table import join, Table
import astropy.units as u
from lsst.afw.table import SourceCatalog
import lsst.gauss2d as g2d
import lsst.gauss2d.fit as g2f
from lsst.meas.extensions.multiprofit.errors import IsParentError, NotPrimaryError
from lsst.multiprofit.fitting import (
    CatalogExposureSourcesABC, CatalogSourceFitterABC,
    CatalogSourceFitterConfig, CatalogSourceFitterConfigData,
)
from lsst.multiprofit.utils import get_params_uniq
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.fit_coadd_multiband import CoaddMultibandFitSubConfig
import numpy as np
import pydantic
from smatch.matcher import Matcher

from .fit_cosmos_hst import CatalogExposureCosmosHstBase, NoHstSourceFoundError, NotBrightHstStarError

__all__ = [
    "CatalogExposureCosmosHstStars", "MultiProFitCosmosHstStarsConfig", "MultiProFitCosmosHstStarsFitter",
]


class MultiProFitCosmosHstStarsConfig(CatalogSourceFitterConfig, CoaddMultibandFitSubConfig):
    """Configuration for the MultiProFit COSMOS-HST star fitter."""

    def bands_read_only(self) -> set:
        return set()

    mag_column_hst = pexConfig.Field[str](
        default="mag_best",
        doc="Column to use as estimate of HST magnitude for selection",
    )
    mag_minimum_hst = pexConfig.Field[float](
        default=23,
        doc="Faintest allowed HST magnitude to select fit candidates",
    )
    initialize_ellipses = pexConfig.Field[bool](
        default=True,
        doc="Whether to initialize the ellipse parameters from the model config; if False, they "
        "will remain at the best-fit values for the previous source's PSF",
    )
    prefix_column = pexConfig.Field[str](default="mpf_psf_", doc="Column name prefix")
    radius_cutout_pix = pexConfig.Field[int](default=35, doc="Size of the cutout to fit in pixels")

    def setDefaults(self):
        super().setDefaults()
        self.apply_centroid_pixel_offset = False

class CatalogExposureCosmosHstStars(CatalogExposureCosmosHstBase):
    """A catexp for fitting stars to characterize the PSF."""

    model_config: ClassVar = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    config_fit: MultiProFitCosmosHstStarsConfig = pydantic.Field(title="Config for fitting options")

    names_config_table: ClassVar = {
        "size_x": "sigma_x",
        "size_y": "sigma_y",
        "rho": "rho",
    }

    @classmethod
    def adjust_psf_table_values(cls, table_psf_fits: Table, band: str | None = None):
        """Add the minimal model back to best-fit PSF model parameter values.

        Parameters
        ----------
        table_psf_fits
            The table to adjust.
        band
            A band to strip out of Gaussian component flux column names.
        """
        config = table_psf_fits.meta["config"]
        model_minimal = table_psf_fits.meta["config_psf_model_minimal"]
        prefix_column = config["prefix_column"]

        sigma_x_min, sigma_y_min, rho_min = (
            model_minimal[name]["value_initial"] for name in cls.names_config_table.keys()
        )
        sigma_x_sq_min = sigma_x_min**2
        sigma_y_sq_min = sigma_y_min**2
        cov_min = sigma_x_min*sigma_y_min*rho_min
        config_model = config["config_model"]
        group = next(iter(next(iter(config_model["sources"].values()))["component_groups"].values()))
        gaussians = group["components_gauss"]

        columns_flux = []

        for name_comp, config_comp in gaussians.items():
            # It wouldn't make sense to fix only 1-2 of these params, but
            # all three would be reasonable
            column_format = "{prefix}{comp}_{key}"
            key_flux = column_format.format(prefix=prefix_column, comp=name_comp, key=f"flux")
            if band is not None:
                key_old = column_format.format(prefix=prefix_column, comp=name_comp, key=f"{band}_flux")
                table_psf_fits.rename_column(key_old, key_flux)
            columns_flux.append(key_flux)

            sigma_x, sigma_y, rho = (
                table_psf_fits[column_format.format(prefix=prefix_column, comp=name_comp, key=key_table)]
                if not config_comp[name_config]["fixed"]
                else config_comp[name_config]["value_initial"]
                for name_config, key_table in cls.names_config_table.items()
            )
            sigma_x_sq = sigma_x_sq_min + sigma_x**2
            sigma_y_sq = sigma_y_sq_min + sigma_y**2
            cov = cov_min + sigma_x*sigma_y*rho
            sigma_x = np.sqrt(sigma_x_sq)
            sigma_y = np.sqrt(sigma_y_sq)
            rho = cov/(sigma_x*sigma_y)
            for value, (name_config, key_table) in zip(
                (sigma_x, sigma_y, rho),
                cls.names_config_table.items(),
            ):
                if config_comp[name_config]["fixed"]:
                    config_comp[name_config]["value_initial"] = value
                else:
                    name_column = column_format.format(prefix=prefix_column, comp=name_comp, key=key_table)
                    table_psf_fits[name_column] = value

        sum_fluxes = np.sum([table_psf_fits[column] for column in columns_flux], axis=0)

        # Set a floor of 1e-8*total for each component
        for column in columns_flux:
            value_min = 1e-8*sum_fluxes
            low = table_psf_fits[column] < value_min
            table_psf_fits[column][low] = value_min[low]

        # Recalculate sums to ensure normalization
        sum_fluxes = np.sum([table_psf_fits[column] for column in columns_flux], axis=0)
        for column in columns_flux:
            table_psf_fits[column] = table_psf_fits[column]/sum_fluxes

    def get_psf_model(self, params: Mapping[str, Any]) -> g2f.PsfModel | None:
        return self.psf_model_minimal

    def get_source_observation(self, source: Mapping[str, Any], **kwargs: Any) -> g2f.ObservationD | None:
        if not kwargs.get("skip_flags"):
            if (not source["detect_isPrimary"]) or source["merge_peak_sky"]:
                raise NotPrimaryError(f"source {source[self.config_fit.column_id]} has invalid flags for fit")
        ra, dec = source["coord_ra"].asDegrees(), source["coord_dec"].asDegrees()
        ra_hst, dec_hst = (self.catalog_hst[column] for column in ("ra", "dec"))
        max_sep_deg = 1/3600
        within = np.logical_and(*(
            (coord_hst > (cen_source - max_sep_deg)) & (coord_hst < (cen_source + max_sep_deg))
            for coord_hst, cen_source in ((ra_hst, ra), (dec_hst, dec))
        ))
        if not np.any(within):
            raise NoHstSourceFoundError(f"no HST sources found within {max_sep_deg*3600} arcsec")

        catalog_within = self.catalog_hst[within]
        idx_brightest = np.argmax(catalog_within["mag_best"])
        brightest = catalog_within[idx_brightest]
        errors = []
        if brightest["mu_class"] != 2:
            errors.append(f'mu_class={brightest["mu_class"]} (star=2)')
        if not (brightest["mag_auto"] < self.config_fit.mag_minimum_hst):
            errors.append(f'mu_class={brightest["mu_class"]} (star=2)')
        if errors:
            raise NotBrightHstStarError(f'Not an HST bright star due to: {";".join(errors)}')

        indices_matched = self.metadata.get("idx_matched")
        if indices_matched is None:
            indices_matched = {}
            self.metadata["idx_matched"] = indices_matched
        try:
            indices_matched[source["id"]] = brightest["index_row"] if "index_row" in brightest else (
                np.argwhere(within)[idx_brightest][0]
            )
        except:
            arg_within = np.argwhere(within)
            n_within = len(arg_within)

        ra_hst = brightest["ra"]
        dec_hst = brightest["dec"]
        obs_hst = self.observation_hst
        wcs_hst = self.wcs_hst
        img_hst = obs_hst.image.data

        pix_y, pix_x = wcs_hst.world_to_array_index(SkyCoord(ra_hst*u.deg, dec_hst*u.deg))

        size_cutout = self.config_fit.radius_cutout_pix
        xy_begin = [pix_x - size_cutout, pix_y - size_cutout]
        cutout = img_hst[xy_begin[1]:pix_y + size_cutout, xy_begin[0]:pix_x + size_cutout]

        # Find the indices of the nine brightest pixels
        # (a cheaper proxy for a new centroid than something moment-based)
        columns_brightest, rows_brightest = np.unravel_index(
            np.argpartition(cutout, -9, axis=None)[-9:],
            cutout.shape,
        )
        xy_peak = (np.mean(rows_brightest), np.mean(columns_brightest))
        xy_begin = tuple(int(round(begin - size_cutout + peak)) for begin, peak in zip(xy_begin, xy_peak))
        slice_cutout_x = slice(xy_begin[0], xy_begin[0] + 2*size_cutout)
        slice_cutout_y = slice(xy_begin[1], xy_begin[1] + 2*size_cutout)

        dy2 = self.wcs_hst.wcs.cd[1, 1]

        radec_begin = wcs_hst.array_index_to_world(xy_begin[1], xy_begin[0])
        dec_min = radec_begin.dec.value - 0.5*dy2
        cos_dec_avg = math.cos((dec_min + dy2*size_cutout)*math.pi/180.)
        dx1 = self.wcs_hst.wcs.cd[0, 0]/cos_dec_avg

        coordsys = g2d.CoordinateSystem(
            dx1=dx1,
            dy2=dy2,
            x_min=radec_begin.ra.value - 0.5*dx1,
            y_min=dec_min,
        )

        observation = g2f.ObservationD(
            image=g2d.ImageD(img_hst[slice_cutout_y, slice_cutout_x], coordsys=coordsys),
            sigma_inv=g2d.ImageD(obs_hst.sigma_inv.data[slice_cutout_y, slice_cutout_x], coordsys=coordsys),
            mask_inv=g2d.ImageB(obs_hst.mask_inv.data[slice_cutout_y, slice_cutout_x], coordsys=coordsys),
            channel=obs_hst.channel,
        )

        return observation

    @classmethod
    def populate_psf_table(cls, table_psf_fits: Table, catalog_ref: SourceCatalog, band: str | None = None):
        """Copy best-fit PSF parameters from the nearest good fit."""

        prefix = table_psf_fits.meta["config"]["prefix_column"]

        flags = [col for col in table_psf_fits.colnames if col.endswith("_flag")]
        good = ~np.any(
            [table_psf_fits[col] for col in flags],
            axis=0,
        )
        ref = table_psf_fits[good]
        n_match_max = 1

        with Matcher(ref[f"{prefix}cen_x"], ref[f"{prefix}cen_y"]) as matcher:
            idxs_ref = matcher.query_knn(
                catalog_ref["coord_ra"]*180/math.pi,
                catalog_ref["coord_dec"]*180/math.pi,
                distance_upper_bound=1.0,
                k=n_match_max,
            )

        columns = ["cen_x", "cen_y"]
        config_source = next(iter(
            next(iter(
                next(iter(
                    table_psf_fits.meta["config"]["config_model"]["sources"].values()
                )).values()  # list of component groups for the first source
            )).values()  # list of component configs for the first group
        ))  # the actual component config

        for name_gauss, config_component in config_source["components_gauss"].items():
            for column in ("rho", "size_x", "size_y"):
                if not config_component[column]['fixed']:
                    columns.append(f"{name_gauss}_{cls.names_config_table[column]}")
            columns.append(f"{name_gauss}{f'_{band}' if band is not None else ''}_flux")

        for column in columns:
            column_full = f"{prefix}{column}"
            table_psf_fits[column_full] = ref[column_full][idxs_ref]

        for column_full in flags:
            table_psf_fits[column_full] = ref[column_full][idxs_ref]


class MultiProFitCosmosHstStarsFitter(CatalogSourceFitterABC):
    """A MultiProFit source fitter for COSMOS-HST stars.

    Parameters
    ----------
    errors_expected
        A dictionary of exceptions that are expected to sometimes be raised
        during processing (e.g. for missing data) keyed by the name of the
        flag column used to record the failure.
    add_missing_errors
        Whether to add all of the standard MultiProFit errors with default
        column names to errors_expected, if not already present.
    **kwargs
        Keyword arguments to pass to the superclass constructor.
    """

    def __init__(
        self,
        errors_expected: dict[str, Exception] | None = None,
        add_missing_errors: bool = True,
        **kwargs: Any,
    ):
        if errors_expected is None:
            errors_expected = {}
        if add_missing_errors:
            for error_catalog in (
                IsParentError, NoHstSourceFoundError, NotBrightHstStarError, NotPrimaryError,
            ):
                if error_catalog not in errors_expected:
                    errors_expected[error_catalog] = error_catalog.column_name()
        super().__init__(errors_expected=errors_expected, **kwargs)

    def copy_centroid_errors(
        self,
        columns_cenx_err_copy: tuple[str],
        columns_ceny_err_copy: tuple[str],
        results: Table,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
    ):
        for column in columns_cenx_err_copy:
            results[column] = catalog_multi["slot_Centroid_xErr"]
        for column in columns_ceny_err_copy:
            results[column] = catalog_multi["slot_Centroid_yErr"]

    def fit(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> Table:
        catexps_hst = [catexp for catexp in catexps if catexp.band == "F814W"]
        if len(catexps_hst) != 1:
            raise RuntimeError("Did not find exactly one catexp with band='F814W'")
        catexp_hst: CatalogExposureCosmosHstStars = catexps_hst[0]
        config_psf = catexp_hst.psf_model_minimal_config
        config_psf_dict = config_psf.toDict()
        results = super().fit(
            catalog_multi=catalog_multi, catexps=catexps, config_data=config_data, logger=logger, **kwargs
        )
        results.meta["config_psf_model_minimal"] = config_psf_dict
        idx_matched = catexp_hst.metadata["idx_matched"]
        table_matched = Table({"id": np.array(list(idx_matched.keys()), dtype=np.int64),
                               "row_catalog_hst": np.array(list(idx_matched.values()), dtype=np.int32)})
        table_matched = join(results[["id"]], table_matched, join_type="left", keys="id")
        results["row_catalog_hst"] = table_matched["row_catalog_hst"]
        return results

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float):
        return cen_x, cen_y

    def initialize_model(
        self,
        model: g2f.ModelD,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        centroid_pixel_offset: float = 0,
        **kwargs,
    ):
        if values_init is None:
            values_init = {}
        if kwargs:
            raise ValueError(f"Unexpected {kwargs=}")
        # Maybe consider using HST catalog values for something. Or don't
        # catalog_hst = catexps[0].catalog_hst
        observation = model.data[0]

        coordsys = observation.image.coordsys
        x_min, y_min = coordsys.x_min, coordsys.y_min
        x_max = coordsys.x_min + float(observation.image.n_cols)*coordsys.dx1
        if coordsys.dx1 < 0:
            x_min, x_max = x_max, x_min
        y_max = coordsys.y_min + float(observation.image.n_rows)*coordsys.dy2
        if coordsys.dy2 < 0:
            y_min, y_max = y_max, y_min

        cen_x = (x_min + x_max)/2.0
        cen_y = (y_min + y_max)/2.0

        sigma = 0.8*math.sqrt(coordsys.dx1**2 + coordsys.dy2**2)

        # An R_eff larger than the box size is problematic.
        limits_size = max(50.0 * sigma, 2.0 * np.hypot(x_max - x_min, y_max - y_min))
        limits_xy = (sigma/20, limits_size)
        size_ratio = 1.5

        params_limits_init = {
            g2f.CentroidXParameterD: (cen_x, (x_min, x_max), None),
            g2f.CentroidYParameterD: (cen_y, (y_min, y_max), None),
            g2f.ReffXParameterD: (sigma, limits_xy, size_ratio),
            g2f.ReffYParameterD: (sigma, limits_xy, size_ratio),
            g2f.SigmaXParameterD: (sigma, limits_xy, size_ratio),
            g2f.SigmaYParameterD: (sigma, limits_xy, size_ratio),
            g2f.RhoParameterD: (0., None, None),
        }

        source = model.sources[0]
        # TODO: There ought to be a better way to not get the PSF centroids
        # (those are part of model.data's fixed parameters)
        params_init = get_params_uniq(source)
        flux_total = np.sum(observation.image.data)
        flux_init = flux_total/len(source.components)
        limits_flux = (1e-5*flux_total, 10.*flux_total)

        for param in params_init:
            if param.linear:
                value_init = flux_init
                limits_new = limits_flux
                value_init_ratio = None
            else:
                value_init, limits_new, value_init_ratio = params_limits_init.get(
                    type(param),
                    (values_init.get(param), None, None),
                )
            if limits_new:
                param.limits = g2f.LimitsD(limits_new[0], limits_new[1])
            if value_init is not None:
                param.value = value_init
            if value_init_ratio is not None:
                params_limits_init[type(param)] = (value_init*value_init_ratio, limits_new, value_init_ratio)

        for prior in model.priors:
            if isinstance(prior, g2f.GaussianPrior):
                # TODO: Add centroid prior
                pass
            elif isinstance(prior, g2f.ShapePrior):
                # TODO: Consider if it makes any sense to add a size prior
                pass

    def validate_fit_inputs(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData = None,
        logger: logging.Logger = None,
        **kwargs: Any,
    ) -> None:
        errors = []
        for idx, catexp in enumerate(catexps):
            if not isinstance(catexp, CatalogExposureCosmosHstStars):
                errors.append(f"catexps[{idx=} {type(catexp)=} !isinstance(CatalogExposurePsfs)")
        if len(catexps) != 1:
            errors.append(f"{len(catexps)=} != 1")
        if errors:
            raise RuntimeError("\n".join(errors))
