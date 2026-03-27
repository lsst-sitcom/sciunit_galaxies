import logging
import math
from functools import cached_property
from typing import Any, Mapping, Sequence

import astropy
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import lsst.afw.geom
import lsst.gauss2d as g2d
import lsst.gauss2d.fit as g2f
import lsst.pipe.tasks.fit_coadd_multiband as fitMB
from lsst.meas.extensions.multiprofit.fit_coadd_multiband import (
    CatalogExposurePsfs, MultiProFitSourceConfig, MultiProFitSourceFitter,
    CatalogExposureSourcesDataclassConfig, CatalogExposureSourcesWcsBase,
    CachedBasicModelInitializer, MakeCachedBasicInitializerAction, MakeInitializerActionBase,
    ModelInitializer,
)
from lsst.meas.extensions.multiprofit.errors import NotPrimaryError
from lsst.multiprofit.fitting import (
    CatalogExposureSourcesABC,
    CatalogSourceFitterConfigData,
)
from lsst.multiprofit.modeller import Model
import lsst.pex.config as pexConfig
import numpy as np
import pydantic

from .fit_cosmos_hst import CatalogExposureCosmosHstBase
from .fit_cosmos_hst_stars import MultiProFitCosmosHstStarsConfig

__all__ = [
    "CatalogExposureCosmosHstObjects", "MultiProFitCosmosHstObjectsConfig",
    "MultiProFitCosmosHstObjectsFitter",
]


class CachedCosmosHstModelInitializerConfig(pexConfig.Config):
    chained = pexConfig.Field[bool](
        doc="Whether to initialize from previous model fits",
        default=False,
    )
    prefix_cen_hsc = pexConfig.Field[str](
        doc="Prefix for HSC centroid columns",
        default="hsc_",
    )
    prefix_cen_hst = pexConfig.Field[str](
        doc="Prefix for HST centroid columns",
        default="hst_"
    )


class CachedCosmosHstModelInitializer(CachedBasicModelInitializer):
    """A COSMOS HST/HSC initializer."""
    config_cosmos: CachedCosmosHstModelInitializerConfig = pydantic.Field(
        title="COSMOS-specific configuration settings",
    )

    def get_flux_init(self, source: Mapping[str, Any], catexp: CatalogExposureSourcesABC, is_afw: bool):
        # Rely on fallback mechanisms
        # TODO: Consider using catalog?
        if is_afw:
            return super().get_flux_init(source, catexp, True)
        return 1

    def get_centroid_and_shape(
        self,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
        values_init: Mapping[g2f.ParameterD, float] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float, float], bool]:
        if not self.config_cosmos.chained:
            return super().get_centroid_and_shape(
                source=source, catexps=catexps, config_data=config_data, values_init=values_init,
            ) + (False,)
        row_best = None
        chisq_red_min = math.inf
        for name, input_data in self.inputs.items():
            data = input_data.data
            index_row = input_data.id_index.get(source["id"])
            if index_row is not None:
                row = data[index_row]
                chisq_red = input_data.get_column("chisq_reduced", data=row)
                if chisq_red < chisq_red_min:
                    row_best = (row, input_data)
                    chisq_red_min = chisq_red
        found = row_best is not None
        if not found:
            return super().get_centroid_and_shape(
                source=source, catexps=catexps, config_data=config_data, values_init=values_init,
            ) + (False,)
        row_best, input_data = row_best
        cen_x, cen_y, reff_x, reff_y, rho, cen_x_hst, cen_y_hst = (
            input_data.get_column(column, data=row_best, prefix=prefix)
            for column, prefix in (
                ("cen_x", self.config_cosmos.prefix_cen_hsc),
                ("cen_y", self.config_cosmos.prefix_cen_hsc),
                (f"{input_data.size_column}_x", None),
                (f"{input_data.size_column}_y", None),
                ("rho", None),
                ("cen_x", self.config_cosmos.prefix_cen_hst),
                ("cen_y", self.config_cosmos.prefix_cen_hst),
            )
        )
        return (cen_x, cen_y), (reff_x, reff_y, rho), (cen_x_hst, cen_y_hst)

    def initialize_model(
        self,
        model: Model,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        **kwargs,
    ):
        if values_init is None:
            values_init = {}
        set_flux_limits = kwargs.pop("set_flux_limits", True)
        flux_init_min = kwargs.pop("value_init_min", 1e-10)
        flux_limit_min = kwargs.pop("flux_limit_min", 1e-12)
        wcs = kwargs.pop("wcs")
        if kwargs:
            raise ValueError(f"Unexpected {kwargs=}")
        centroid_pixel_offset = config_data.config.centroid_pixel_offset

        # Make restrictive centroid limits (intersection, not union)
        x_min, y_min, x_max, y_max = -np.inf, -np.inf, np.inf, np.inf

        fluxes_init = {}
        fluxes_limits = {}

        # This is the maximum number of potential observations
        # They might not all have made it into the data
        n_catexps = len(catexps)
        n_components = len(model.sources[0].components)

        # If not true, some bands must have no data to fit
        if len(catexps) != len(model.data):
            catexps_obs = []
            for catexp in catexps:
                fluxes_init[catexp.channel] = flux_init_min
                fluxes_limits[catexp.channel] = (0, np.inf)
                # No associated catalog means we can't fit (and should be
                # because there's no exposure for this band in this patch)
                if len(catexp.get_catalog()) > 0:
                    catexps_obs.append(catexp)
        else:
            catexps_obs = catexps

        (cen_x, cen_y), (sig_x_pix, sig_y_pix, rho_pix), cens_hst = self.get_centroid_and_shape(
            source,
            catexps,
            config_data,
            values_init=values_init,
        )
        has_radec_hst = cens_hst is not None and np.isfinite([cens_hst[0], cens_hst[1]]).all()

        ra, dec, sig_ra, sig_dec, rho_radec = self.convert_coordinates(
            None if has_radec_hst else wcs, cen_x, cen_y, sig_x_pix, sig_y_pix, rho_pix,
        )
        if has_radec_hst:
            ra, dec = cen_x, cen_y
            ra_hst, dec_hst = cens_hst
        else:
            ra_hst, dec_hst = None, None

        for idx_obs, observation in enumerate(model.data):
            coordsys = observation.image.coordsys
            catexp: CatalogExposureSourcesWcsBase = catexps_obs[idx_obs]
            band = catexp.band
            if not isinstance(catexp, CatalogExposureSourcesWcsBase):
                raise ValueError(f"catexps[{band}] must be a CatalogExposureSourcesWcsABC")

            is_afw = isinstance(catexp, CatalogExposurePsfs)

            x_coordsys = (
                coordsys.x_min,
                coordsys.x_min + float(observation.image.n_cols) * coordsys.dx1,
            )
            y_coordsys = (
                coordsys.y_min,
                coordsys.y_min + float(observation.image.n_rows) * coordsys.dy2,
            )

            x_min = max(x_min, min(x_coordsys))
            y_min = max(y_min, min(y_coordsys))
            x_max = min(x_max, max(x_coordsys))
            y_max = min(y_max, max(y_coordsys))

            if (not has_radec_hst) and (not is_afw) and (
                (catalog_hst := getattr(catexp, "catalog_hst", None)) is not None
            ):
                within_ra = (catalog_hst["ra"] > x_min) & (catalog_hst["ra"] < x_max)
                within_dec = (catalog_hst["dec"] > y_min) & (catalog_hst["dec"] < y_max)
                within_both = within_ra & within_dec
                n_within = np.sum(within_both)
                if n_within >= 1:
                    ra_hst, dec_hst = catalog_hst["ra"][within_both], catalog_hst["dec"][within_both]
                    if n_within == 1:
                        ra_hst, dec_hst = ra_hst[0], dec_hst[0]
                    else:
                        dists = np.hypot(ra_hst - ra, dec_hst - dec)
                        dist_min = np.argmin(dists)
                        ra_hst, dec_hst = ra_hst[dist_min], dec_hst[dist_min]
                    has_radec_hst = True

            flux_total = np.nansum(observation.image.data[observation.mask_inv.data])
            flux_init = self.get_flux_init(source, catexp, is_afw=is_afw)

            flux_init = flux_init if (flux_init > 0) else max(flux_total, 1.0)
            if set_flux_limits:
                flux_max = 10 * max((flux_init, flux_total))
                flux_min = min(flux_limit_min, flux_max / 1000)
            else:
                flux_min, flux_max = 0, np.inf
            if not (flux_init > flux_min):
                flux_upper = flux_max if (flux_max < np.inf) else 10. * flux_min
                flux_init = flux_min + 0.01 * (flux_upper - flux_min)
            fluxes_init[observation.channel] = flux_init / n_components
            fluxes_limits[observation.channel] = (flux_min, flux_max)

        # If we couldn't get a shape at all, make it small and roundish
        if not np.isfinite(rho_radec):
            # Note rho=0 (circular) is generally disfavoured by shape priors
            # However, setting it to a non-zero value seems to make scipy
            # fail to move off initial conditions, as do sizes below 2 pixels
            sig_ra, sig_dec, rho_radec = 1.0/3600, 1.0/3600, 0.0

        # An R_eff larger than the box size is problematic. This should also
        # stop unreasonable size proposals; a log10 transform isn't enough.
        # TODO: Try logit for r_eff?
        size_major = g2d.EllipseMajor(g2d.Ellipse(sigma_x=sig_ra, sigma_y=sig_dec, rho=rho_radec)).r_major
        limits_size = max(
            5.0 * size_major,
            2.0 * np.hypot((x_max - x_min)/math.cos(dec*math.pi/180), y_max - y_min),
        )
        limits_xy = (1e-4/3600, limits_size)
        params_limits_init = {
            g2f.CentroidXParameterD: {
                "hst": ((ra_hst if has_radec_hst is not None else ra), (x_min, x_max)),
                "": (ra, (x_min, x_max)),
            },
            g2f.CentroidYParameterD: {
                "hst": ((dec_hst if has_radec_hst is not None else dec), (y_min, y_max)),
                "": (dec, (y_min, y_max)),
            },
            g2f.ReffXParameterD: {"": (sig_ra, limits_xy)},
            g2f.ReffYParameterD: {"": (sig_dec, limits_xy)},
            g2f.SigmaXParameterD: {"": (sig_ra, limits_xy)},
            g2f.SigmaYParameterD: {"": (sig_dec, limits_xy)},
            g2f.RhoParameterD: {"": (rho_radec, None)},
            # TODO: get guess from configs?
            g2f.SersicMixComponentIndexParameterD: {"": (1.0, None)},
        }

        fluxes_init_tuple = tuple(fluxes_init.values())
        fluxes_limits_tuple = tuple(fluxes_limits.values())
        idx_obs = 0
        for param in self.params_init:
            if param.linear:
                value_init = fluxes_init_tuple[idx_obs]
                limits_new = fluxes_limits_tuple[idx_obs]
                idx_obs += 1
                if idx_obs == n_catexps:
                    idx_obs = 0
            else:
                type_param = type(param)
                limits_default = params_limits_init.get(type_param)
                limits_fallback = (values_init.get(param), None)
                value_init, limits_new = (
                    limits_default.get(param.label, limits_default.get("", limits_fallback))
                    if limits_default is not None
                    else limits_fallback
                )
            if limits_new:
                param.limits = g2f.LimitsD(limits_new[0], limits_new[1])
            if value_init is not None:
                param.value = value_init

        priors_shape_mag = self.priors_shape_mag
        has_priors_mag = len(priors_shape_mag) > 0
        if has_priors_mag:
            mag_total = u.nJy.to(u.ABmag, np.nansum(fluxes_init_tuple))

        # TODO: Add centroid prior
        priors_gauss, priors_shape = self.get_priors_type(model)
        for prior in priors_shape:
            if has_priors_mag and ((prior_adjustments := priors_shape_mag.get(prior)) is not None):
                mag_dep_prior, prior_shape_new = prior_adjustments
                prior_size_new = prior_shape_new.prior_size
                # the size-apparent mag relation probably flattens
                # for very bright/faint objects - maybe not so
                # sharply, but clipping a broad mag range ought to be fine
                prior.prior_size.mean_parameter.value = prior_size_new.mean_parameter.value * 10 ** (
                    mag_dep_prior.slope_median_per_mag * np.clip(
                        mag_total - mag_dep_prior.intercept_mag,
                        -12.5,
                        12.5,
                    )
                )/3600.
                # Note the /3600 above is because the size is in degrees, not arcsec

                # it's uncertain how the intrinsic scatter behaves
                # educated guess is it doesn't change much, also
                # one runs out of bright galaxies to measure it anyway
                prior.prior_size.stddev_parameter.value = prior_size_new.stddev_parameter.value * 10 ** (
                    mag_dep_prior.slope_stddev_per_mag * np.clip(
                        mag_total - mag_dep_prior.intercept_mag,
                        -12.5,
                        12.5,
                    )
                )
            else:
                prior.prior_size.mean_parameter.value = size_major


class MakeCachedCosmosHstInitializerAction(MakeCachedBasicInitializerAction):
    config_cosmos = pexConfig.ConfigField[CachedCosmosHstModelInitializerConfig](
        doc="COSMOS-specific initializer config settings",
    )

    def _make_initializer(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        config_data: CatalogSourceFitterConfigData,
    ) -> ModelInitializer:
        sources, priors = config_data.sources_priors
        return CachedCosmosHstModelInitializer(
            config=self.config, priors=priors, sources=sources, config_cosmos=self.config_cosmos,
        )


class MultiProFitCosmosHstObjectsConfig(MultiProFitSourceConfig):
    """Configuration for the MultiProFit COSMOS-HST object fitter."""

    psf_sigma_subtract_hst = pexConfig.Field[float](
        doc="PSF sigma (deg) to subtract in quadrature from best-fit HST values",
        default=0,
        check=lambda x: np.isfinite(x) and (x >= 0),
    )

    @staticmethod
    def get_default_action_initializer() -> MakeInitializerActionBase:
        action_initializer = MakeCachedCosmosHstInitializerAction()
        action_initializer.config.rho_abs_max = 0.9
        return action_initializer

    def setDefaults(self):
        super().setDefaults()
        self.action_initializer = self.get_default_action_initializer()


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, config=CatalogExposureSourcesDataclassConfig)
class CatalogExposureCosmosHstObjects(CatalogExposureCosmosHstBase, CatalogExposureSourcesWcsBase):
    """A catexp for fitting stars to characterize the PSF."""

    table_psf_fits: Table = pydantic.Field(title="Table of PSF fit parameters")
    wcs_ref: lsst.afw.geom.SkyWcs = pydantic.Field(title="The WCS for the reference catalog")

    def get_psf_model(self, params: Mapping[str, Any]) -> g2f.PsfModel | None:
        return super().get_psf_model(params)
        match = np.argwhere(
            self.table_psf_fits[self.psf_model_data.config.column_id] == params[self.config_fit.column_id]
        )[0][0]
        self.psf_model_data.init_psf_model(self.table_psf_fits[match])
        return self.psf_model_data.psf_model

    @cached_property
    def psf_sigma_subtract(self) -> float:
        return self.config_fit.psf_sigma_subtract_hst

    def get_source_observation(self, source: Mapping[str, Any], **kwargs: Any) -> g2f.ObservationD | None:
        if not kwargs.get("skip_flags"):
            if (not source["detect_isPrimary"]) or source["merge_peak_sky"]:
                raise NotPrimaryError(f"source {source[self.config_fit.column_id]} has invalid flags for fit")
        plot = kwargs.pop("plot", False)

        # ra, dec = source["coord_ra"].asDegrees(), source["coord_dec"].asDegrees()

        footprint = source.getFootprint()
        bbox = footprint.getBBox()
        radec_begin = self.wcs_ref.pixelToSky(bbox.beginX, bbox.beginY)
        radec_end = self.wcs_ref.pixelToSky(bbox.endX, bbox.endY)

        obs_hst = self.observation_hst
        wcs_hst = self.wcs_hst
        img_hst = obs_hst.image.data

        yx_begin, yx_end = (
            wcs_hst.world_to_array_index(
                SkyCoord(radec.getRa().asDegrees()*u.deg, radec.getDec().asDegrees()*u.deg)
            )
            for radec in (radec_begin, radec_end)
        )

        slice_cutout_x = slice(yx_begin[1], yx_end[1])
        slice_cutout_y = slice(yx_begin[0], yx_end[0])

        wcs_cd = self.wcs.get_cd_matrix()
        dx1 = wcs_cd[0, 0]
        dy2 = wcs_cd[1, 1]

        radec_begin = wcs_hst.array_index_to_world(yx_begin[0], yx_begin[1])
        dec_min = radec_begin.dec.value - 0.5*dy2

        # TODO: We should really fit RA*cos(dec) ~= x, not just RA
        # cos_dec_avg = math.cos((dec_min + dy2*(yx_end[0] - yx_begin[0]))*math.pi/180.)

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

        if plot:
            import matplotlib.pyplot as plt
            plt.imshow(np.log10(observation.image.data))
            plt.show(block=False)

        return observation


class MultiProFitCosmosHstObjectsFitter(MultiProFitSourceFitter):
    """A MultiProFit source fitter for COSMOS joint fitting."""

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float):
        return cen_x, cen_y

    def post_fit(
        self,
        idx: int,
        model: g2f.ModelF | g2f.ModelD,
        results: astropy.table.row.Row,
        params_cen_x: dict[str, g2f.CentroidXParameterD],
        params_cen_y: dict[str, g2f.CentroidYParameterD],
    ):
        debug = False
        if debug:
            i_cen_x = iter(params_cen_x.values())
            i_cen_y = iter(params_cen_y.values())
            dist = np.hypot(
                next(i_cen_x).value - next(i_cen_x).value,
                next(i_cen_y).value - next(i_cen_y).value,
            )
            if dist > 4e-14:
                import matplotlib.pyplot as plt
                from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
                from lsst.multiprofit.plotting import plot_model_rgb, plot_model_singleband

                model_eval = g2f.ModelD(
                    data=model.data, psfmodels=model.psfmodels, sources=model.sources
                )
                model_eval.setup_evaluators(evaluatormode=g2f.EvaluatorMode.image)
                model_eval.evaluate()

                coordsyses = {
                    "hsc": model.data[0].image.coordsys,
                    "hst": model.data[len(model.data) - 1].image.coordsys,
                }

                (cx_hsc_init, cx_hsc), (cx_hst_init, cx_hst) = (
                    (
                        (cx - coordsys.x_min) / coordsys.dx1
                        for cx in (results[f"{surv}_cen_x_init"], results[f"{surv}_cen_x"])
                    ) for surv, coordsys in coordsyses.items()
                )
                (cy_hsc_init, cy_hsc), (cy_hst_init, cy_hst) = (
                    (
                        (cy - coordsys.y_min) / coordsys.dy2
                        for cy in (results[f"{surv}_cen_y_init"], results[f"{surv}_cen_y"])
                    ) for surv, coordsys in coordsyses.items()
                )

                plot_model_rgb(model, weights={band: bands_weights_lsst[band] for band in "irg"})
                plot_model_singleband(model, 3)

                plt.show()

    def validate_fit_inputs(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData = None,
        logger: logging.Logger = None,
        **kwargs: Any,
    ) -> None:
        errors = []
        bands = {}
        catexps_fit_cmb = []
        for idx, catexp in enumerate(catexps):
            if isinstance(catexp, CatalogExposurePsfs):
                catexps_fit_cmb.append(catexp)
            if catexp.band in bands:
                errors.append(f"{catexp.band=} {idx=} already in {bands=})")
        if len(catexps) == 0:
            errors.append(f"{len(catexps)=} !> 0")
        try:
            if catexps_fit_cmb:
                super().validate_fit_inputs(
                    catalog_multi=catalog_multi, catexps=catexps_fit_cmb, config_data=config_data,
                    logger=logger, **kwargs
                )
        except Exception as e:
            errors.append(f"super().init() with {len(catexps_fit_cmb)=} got {e}")
        if errors:
            raise RuntimeError("\n".join(errors))
