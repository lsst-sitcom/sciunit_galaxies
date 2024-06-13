# This file is part of sciunit_galaxies.
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

import astropy.io.fits as fits
from astropy.table import Table
from astropy.wcs import WCS
import glob
import gauss2d as g2
import gauss2d.fit as g2f
from itertools import chain
import numpy as np
import pydantic
from typing import ClassVar, Iterable


class CosmosTile(pydantic.BaseModel):
    """A class to load data from COSMOS HST tiles."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    channel: ClassVar[g2f.Channel.get] = g2f.Channel.get("F814W")

    image: fits.hdu.image.PrimaryHDU = pydantic.Field(doc="The image data")
    wcs: WCS = pydantic.Field(doc="The world coordinate system for the image")
    weight: fits.hdu.image.PrimaryHDU = pydantic.Field(doc="The weight (inverse variance) data")

    @classmethod
    def from_tile_table(cls, tile_name: str, tile_table: Table):
        row = tile_table[tile_table["tilename"] == tile_name][0]
        image = fits.open(row["path_science"])[0]
        wcs = WCS(image)
        weight = fits.open(row["path_weight"])[0]
        return cls(image=image, wcs=wcs, weight=weight)

    def make_observation(self) -> g2f.Observation:
        return g2f.Observation(
            channel=self.channel,
            image=g2.ImageD(),
        )


class CosmosTileTable(pydantic.BaseModel):
    """A mini-butler for loading data from COSMOS HST tiles."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    tile_table: Table = pydantic.Field(doc="A tile table as returned by make_tile_table")

    def find_tile_names(self, ra: float, dec: float):
        found = (ra >= self.tile_table["ra_min"]) & (ra <= self.tile_table["ra_max"]) & (
            dec >= self.tile_table["dec_min"]) & (dec <= self.tile_table["dec_max"])
        return self.tile_table["tilename"][found]

    def make_tile(self, tile_name: str):
        return CosmosTile.from_tile_table(tile_name, tile_table=self.tile_table)

    @staticmethod
    def make_tile_table(paths: Iterable[str] | None = None, raise_not_found=True):
        """Get fits objects and skyboxes for all of the HST COSMOS tiles.

        Parameters
        ----------
        paths
            A list of paths containing COSMOS FITS tiles.
        raise_not_found
            Whether to raise if the science file is not found for a given
            weight file; otherwise, the science filename will be left empty.

        Returns
        -------
        table
            A table containing names and corners for each tile.

        Notes
        -----
        The COSMOS data can be downloaded from IRSA at:
         https://irsa.ipac.caltech.edu/data/COSMOS/images/.
        The acs_mosaic_2.0 files are warped to align N-S W-E, whereas the
        unrotated files in acs_2.0 are approximately 11 degrees off clockwise.
        """

        if paths is None:
            paths = ["/sdf/group/rubin/ncsa-project/project/sr525/hstCosmosImages/tiles/"]

        filenames = tuple(sorted(chain.from_iterable(
            glob.glob(f"{path}/acs_I_030mas_*_wht.fits*") for path in paths
        )))
        n_files = len(filenames)
        len_max = max(len(filename) for filename in filenames)
        str_max = " "*(5 + len_max)

        data = {
            "index": np.arange(n_files, dtype=int),
            "tilename": [" "*4]*n_files,
            "path_science": [str_max]*n_files,
            "path_weight": [str_max]*n_files,
        }
        name_coords = ("ra_min", "ra_max", "dec_min", "dec_max")
        for name in name_coords:
            data[name] = np.zeros(n_files, dtype=float)
        data = Table(data)
        n_cut = 0
        idx = 0

        for filename in filenames:
            filename_split = filename.split("_wht.fits")
            glob_sci = f"{filename_split[0]}_sci.fits*"
            filenames_sci = tuple(glob.glob(glob_sci))
            if len(filenames_sci) != 1:
                if raise_not_found:
                    raise RuntimeError(f"Found n!=1 {filenames_sci=} from {filename=} and {glob_sci=}")
                else:
                    filename_sci = ""
            else:
                filename_sci = filenames_sci[0]
            tilename = filename.split("_")[-2]
            idx_tile = np.where(data["tilename"] == tilename)[0]
            if len(idx_tile) > 0:
                assert len(idx_tile) == 1
                idx_tile = idx_tile[0]
                data[idx:]["index"] -= 1
                n_cut += 1
                if filename_sci and not (data["path_science"][idx_tile]):
                    idx_fill = idx_tile
                    idx -= 1
                else:
                    continue
            else:
                idx_fill = idx
            data[idx_fill]["path_science"] = filename_sci
            data[idx_fill]["path_weight"] = filename
            data[idx_fill]["tilename"] = tilename
            image = fits.open(filename)
            wcs = WCS(image[0])
            corners = wcs.calc_footprint()
            for column, coord in enumerate(("ra", "dec")):
                values = corners[:, column]
                data[idx_fill][f"{coord}_min"] = np.min(values)
                data[idx_fill][f"{coord}_max"] = np.max(values)
            idx += 1

        print(idx, n_cut, n_files)
        assert (idx + n_cut) == n_files
        if n_cut:
            data = data[:-n_cut]

        return data
