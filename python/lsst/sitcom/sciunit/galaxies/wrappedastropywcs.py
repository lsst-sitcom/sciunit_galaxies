# This file is part of meas_extensions_multiprofit.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ("WrappedAstropyWcs",)

from functools import cached_property
from typing import Any

import astropy.units as u
import astropy.wcs
import numpy as np
import pydantic

from lsst.meas.extensions.multiprofit.wrappedwcsbase import WrappedWcsBase


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True,
                                config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class WrappedAstropyWcs(WrappedWcsBase):
    """Wrapper for astropy WCS"""
    wcs: astropy.wcs.WCSBase = pydantic.Field(title="The WCS to wrap")

    @cached_property
    def cd_matrix(self) -> np.ndarray:
        return self.wcs.wcs.cd

    def get_cd_matrix(self) -> np.ndarray:
        return self.cd_matrix

    def get_wcs(self) -> Any:
        return self.wcs

    def get_pixel_to_ra_dec(self, x: float, y: float) -> tuple[float, float]:
        coord = self.wcs.pixel_to_world(x, y)
        return tuple(c.to(u.deg) for c in (coord.ra, coord.dec))
