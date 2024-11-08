import warnings
from functools import cached_property
import math
from typing import Any, ClassVar, Iterable

from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from lsst.afw.table import SourceCatalog
import lsst.gauss2d as g2d
import lsst.gauss2d.fit as g2f
from lsst.meas.extensions.multiprofit.errors import CatalogError
from lsst.meas.extensions.multiprofit.fit_coadd_multiband import MultiProFitSourceConfig
from lsst.multiprofit.componentconfig import (
    GaussianComponentConfig,
    ParameterConfig,
)
from lsst.multiprofit.fitting import CatalogExposureSourcesABC, CatalogSourceFitterConfigData
from lsst.multiprofit.utils import get_params_uniq
import numpy as np
import pydantic

__all__ = [
    "NoHstSourceFoundError", "NotBrightHstStarError", "CatalogExposureCosmosHstBase",
]


class NoHstSourceFoundError(CatalogError):
    """RuntimeError for objects that have no nearby HST sources."""

    @classmethod
    def column_name(cls) -> str:
        return "no_hst_source_found_flag"


class NotBrightHstStarError(CatalogError):
    """RuntimeError for objects that have no nearby bright HST stars."""

    @classmethod
    def column_name(cls) -> str:
        return "not_hst_bright_star_flag"


class CatalogExposureCosmosHstBase(CatalogExposureSourcesABC, pydantic.BaseModel):
    """A class to store a catalog, exposure, and metadata for a given dataId.

    The intent is to store an exposure and an associated measurement catalog.
    Users may omit one but not both (e.g. if the intent is just to attach
    a dataId and metadata to a catalog or exposure).
    """

    model_config: ClassVar = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    catalog_hst: Table = pydantic.Field(title="The HST source catalog")
    catalog_ref_hsc: SourceCatalog = pydantic.Field(title="The HSC reference catalog")
    config_fit: MultiProFitSourceConfig = pydantic.Field(title="Config for fitting options")
    cos_dec_hst: float = pydantic.Field(title="The cosine of the average declination of the HST observation")
    dataId: dict = pydantic.Field(title="A DataCoordinate or dict containing a 'band' item")
    id_tract_patch: int = pydantic.Field(0, title="A unique ID for this tract-patch pair")
    metadata: dict = pydantic.Field(default_factory=dict, title="Arbitrary metadata")
    observation_hst: g2f.ObservationD | g2f.ObservationF = pydantic.Field(title="The HST observation")
    wcs_hst: WCS = pydantic.Field(title="The WCS for the HST observation")

    @cached_property
    def band(self) -> str:
        return self.dataId['band']

    @cached_property
    def channel(self) -> g2f.Channel:
        return g2f.Channel.get(self.band)

    def get_catalog(self) -> Iterable:
        return self.catalog_ref_hsc

    # TODO: Implement stitching multiple image/weight pairs into a single
    # patch if needed
    @staticmethod
    def get_observation_hst(
        tiles_within: tuple[str, str],
        radec_min: SkyCoord | None = None,
        radec_max: SkyCoord | None = None,
        band: str = "F814W",
    ) -> tuple[g2f.ObservationD, WCS, float]:
        """Get a gauss2d_fit observation from HST COSMOS image and weight HDUs."""

        # Get the min/max from available tiles
        radec_min_tiles = (360., 90.)
        radec_max_tiles = (0., -90.)
        for filename_image, _ in tiles_within:
            image = fits.open(filename_image)[0]
            wcs = WCS(image)
            shape = image.data.shape
            for x in (-0.5, shape[1] - 0.5):
                for y in (-0.5, shape[0] - 0.5):
                    radec = wcs.pixel_to_world(x, y)
                    radec_min_tiles = (
                        min(radec_min_tiles[0], radec.ra.to(u.deg).value),
                        min(radec_min_tiles[1], radec.dec.to(u.deg).value),
                    )
                    radec_max_tiles = (
                        max(radec_max_tiles[0], radec.ra.to(u.deg).value),
                        max(radec_max_tiles[1], radec.dec.to(u.deg).value),
                    )
        for name, op, radec, radec_tiles, expected in (
            ("min", "<", radec_min, radec_min_tiles, False),
            ("max", ">", radec_max, radec_max_tiles, True),
        ):
            if radec is not None:
                for idx_coord, coord in enumerate(("ra", "dec")):
                    if (getattr(radec, coord).value < radec_tiles[idx_coord]) != expected:
                        name_var = f"{coord}_{name}"
                        warnings.warn(
                            f"requested {name_var}={radec_min[0]} {op} {name_var}_tiles={radec_min_tiles[0]}"
                        )
                        radec.setattr(coord, radec_tiles[idx_coord]*u.deg)

        wcs_obs = WCS(fits.open(filename_image)[0])
        ra_flipped = wcs_obs.wcs.cd[0, 0] < 0
        yx_min = list(wcs_obs.world_to_array_index(radec_min))
        yx_max = list(wcs_obs.world_to_array_index(radec_max))
        if ra_flipped:
            yx_min[1], yx_max[1] = yx_max[1], yx_min[1]
        shape = (yx_max[0] - yx_min[0], yx_max[1] - yx_min[1])

        # reset the reference pixel to the bottom-left corner
        wcs_obs.wcs.crpix = [0.5, 0.5]
        wcs_obs.wcs.crval = [(radec_max if ra_flipped else radec_min).ra.value, radec_min.dec.value]

        image = np.full(shape, fill_value=np.nan, dtype=np.float32)
        sigma_inv = np.full(shape, fill_value=0, dtype=np.float32)

        for filename_image, filename_weight in tiles_within:
            fits_image = fits.open(filename_image)
            fits_weight = fits.open(filename_weight)
            image_iter = fits_image[0]
            wcs = WCS(image_iter)

            image_iter = image_iter.data
            weight_iter = fits_weight[0].data

            yx_begin, yx_end = (list(wcs.world_to_array_index(radec)) for radec in (radec_min, radec_max))
            if ra_flipped:
                yx_begin[1], yx_end[1] = yx_end[1], yx_begin[1]
            yx_min = tuple(x for x in yx_begin)
            yx_begin[0] = max(yx_begin[0], 0)
            yx_begin[1] = max(yx_begin[1], 0)
            yx_end[0] = min(yx_end[0], image_iter.shape[1])
            yx_end[1] = min(yx_end[1], image_iter.shape[0])

            if not ((yx_end[0] > yx_begin[0]) and (yx_end[1] > yx_begin[1])):
                raise ValueError(f"{radec_min=} and {radec_max=} cropped too"
                                 f" aggressively; no pixels are left")
            slices = tuple(slice(yx_begin[dim] - yx_min[dim], yx_end[dim] - yx_min[dim]) for dim in range(2))
            slices_iter = tuple(slice(yx_begin[dim], yx_end[dim]) for dim in range(2))
            image[slices[0], slices[1]] = image_iter[slices_iter[0], slices_iter[1]]
            sigma_inv[slices[0], slices[1]] = np.sqrt(weight_iter[slices_iter[0], slices_iter[1]])

        sigma_inv[~np.isfinite(sigma_inv)] = 0
        mask_inv = sigma_inv > 0

        cos_dec = math.cos((radec_min.dec.value + radec_max.dec.value)*math.pi/360)

        coordsys = g2d.CoordinateSystem(
            dx1=wcs_obs.wcs.cd[0, 0]/cos_dec, dy2=wcs_obs.wcs.cd[1, 1],
            x_min=(radec_max if ra_flipped else radec_min).ra.value,
            y_min=radec_min.dec.value,
        )

        observation = g2f.ObservationD(
            image=g2d.ImageD(image, coordsys=coordsys),
            sigma_inv=g2d.ImageD(sigma_inv, coordsys=coordsys),
            mask_inv=g2d.ImageB(mask_inv, coordsys=coordsys),
            channel=g2f.Channel.get(band),
        )
        return observation, wcs_obs, cos_dec

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        if 'band' not in self.dataId:
            raise ValueError(f"dataId={self.dataId} must have a band")

    @cached_property
    def psf_model_minimal_config(self) -> GaussianComponentConfig:
        coordsys = self.observation_hst.image.coordsys
        dx = abs(coordsys.dx1)
        dy = abs(coordsys.dy2)
        config = GaussianComponentConfig(
            size_x=ParameterConfig(value_initial=0.8*dx, fixed=True),
            size_y=ParameterConfig(value_initial=0.8*dy, fixed=True),
        )
        config.flux.fixed = True
        config.fluxfrac.fixed = True
        config.rho.fixed = True
        config.freeze()
        return config

    @cached_property
    def psf_model_minimal(self) -> g2f.PsfModel:
        """Get a 'minimal' PSF model that ensures that no components will be
           badly unresolved (the maximum difference in the PSF integral with
           sigma_x=sigma_y=0.8, rho=0 is 1.3e-5).
        """
        config = self.psf_model_minimal_config
        component = config.make_component(
            centroid=None,
            integral_model=config.make_linear_integral_model(fluxes={g2f.Channel.NONE: 1.0}),
        ).component
        for param in get_params_uniq(component):
            param.fixed = True
        return g2f.PsfModel([component])


class CatalogSourceFitterCosmosConfigData(CatalogSourceFitterConfigData):
    """Configuration data for a fitter that can initialize lsst.gauss2d.fit
    models with band-dependent, prior-constrained centroids.
    """

    @cached_property
    def sources_priors(self) -> tuple[tuple[g2f.Source], tuple[g2f.Prior]]:

        sources, priors = self.config.make_sources(channels=self.channels)
        return tuple(sources), tuple(priors)
