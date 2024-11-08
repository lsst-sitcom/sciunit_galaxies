import logging
import math
from typing import Any, ClassVar, Mapping, Sequence

from astropy.coordinates import SkyCoord
from astropy.table import join, Table
import astropy.units as u
import lsst.afw.geom
import lsst.gauss2d as g2d
import lsst.gauss2d.fit as g2f
from lsst.meas.extensions.multiprofit.fit_coadd_multiband import (
    CatalogExposurePsfs, MultiProFitSourceConfig, MultiProFitSourceFitter,
)
from lsst.meas.extensions.multiprofit.errors import NotPrimaryError
from lsst.multiprofit.fitting import (
    CatalogExposureSourcesABC,
    CatalogPsfFitterConfig,
    CatalogPsfFitterConfigData,
    CatalogSourceFitterConfigData,
)
from lsst.multiprofit.utils import get_params_uniq, set_config_from_dict
import numpy as np
import pydantic

from .fit_cosmos_hst import CatalogExposureCosmosHstBase
from .fit_cosmos_hst_stars import MultiProFitCosmosHstStarsConfig

__all__ = [
    "CatalogExposureCosmosHstObjects", "MultiProFitCosmosHstObjectsConfig",
    "MultiProFitCosmosHstObjectsFitter",
]


class MultiProFitCosmosHstObjectsConfig(MultiProFitSourceConfig):
    """Configuration for the MultiProFit COSMOS-HST object fitter."""


class CatalogExposureCosmosHstObjects(CatalogExposureCosmosHstBase):
    """A catexp for fitting stars to characterize the PSF."""

    model_config: ClassVar = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    table_psf_fits: Table = pydantic.Field(title="Table of PSF fit parameters")
    wcs_ref: lsst.afw.geom.SkyWcs = pydantic.Field(title="The WCS for the reference catalog")

    def _get_dx1(self):
        return self.wcs_hst.wcs.cd[0, 0]

    def _get_dy2(self):
        return self.wcs_hst.wcs.cd[1, 1]

    def get_psf_model(self, params: Mapping[str, Any]) -> g2f.PsfModel | None:
        match = np.argwhere(
            self.table_psf_fits[self.psf_model_data.config.column_id] == params[self.config_fit.column_id]
        )[0][0]
        self.psf_model_data.init_psf_model(self.table_psf_fits[match])
        return self.psf_model_data.psf_model

    def get_source_observation(self, source: Mapping[str, Any], **kwargs: Any) -> g2f.ObservationD | None:
        if not kwargs.get("skip_flags"):
            if (not source["detect_isPrimary"]) or source["merge_peak_sky"]:
                raise NotPrimaryError(f"source {source[self.config_fit.column_id]} has invalid flags for fit")

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

        dy2 = self._get_dy2()

        radec_begin = wcs_hst.array_index_to_world(yx_begin[0], yx_begin[1])
        dec_min = radec_begin.dec.value - 0.5*dy2

        # TODO: We should really fit RA*cos(dec) ~= x, not just RA
        cos_dec_avg = math.cos((dec_min + dy2*(yx_end[0] - yx_begin[0]))*math.pi/180.)
        dx1 = self._get_dx1()

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

        plot = False
        if plot:
            import matplotlib.pyplot as plt
            plt.imshow(np.log10(observation.image.data))
            plt.show(block=False)

        return observation

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        # TODO: Can/should this be the derived type (MultiProFitPsfConfig)?
        config_dict = self.table_psf_fits.meta["config"]
        config = MultiProFitCosmosHstStarsConfig()
        set_config_from_dict(config, config_dict, initialize_none=True)
        config_psf = CatalogPsfFitterConfig(model=next(iter(config.config_model.sources.values())))
        config_data = CatalogPsfFitterConfigData(config=config_psf)
        object.__setattr__(self, "psf_model_data", config_data)


class MultiProFitCosmosHstObjectsFitter(MultiProFitSourceFitter):
    """A MultiProFit source fitter for COSMOS joint fitting."""

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float):
        return cen_x, cen_y

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
                if not catexp.use_sky_coords:
                    raise ValueError(f"{catexp=} must have use_sky_coords set")
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
