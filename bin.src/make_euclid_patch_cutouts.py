import argparse
import logging

import lsst.daf.butler as dafButler
import lsst.skymap as skymap

from astropy.table import Table
from lsst.sitcom.sciunit.galaxies.euclid import make_patch_cutouts

if __name__ == '__main__':
    parser = argparse.ArgumentParser("make_patch_cutouts")
    parser.add_argument("--band", help="Name of the filter/band", type=str, default=None)
    parser.add_argument("--image_filepath", help="Filepath with Euclid images to make patch cutouts from.", type=str)
    parser.add_argument("--log_level", help="Logging level", type=str, default="INFO")
    parser.add_argument("--mosaics_table_filepath", help="Filepath of Euclid mosaic table",type=str)
    parser.add_argument("--repo", help="Butler repo to load skymap from", type=str, default="/repo/main")
    parser.add_argument("--save_directory", help="Directory in which patch cutouts will be saved.", type=str)
    parser.add_argument("--skymap_collections", help="Collections to query for skymap", type=str, default="skymaps")
    parser.add_argument("--skymap_name", help="Name of the skymap", type=str, default="lsst_cells_v1")
    parser.add_argument("--tracts", help="Comma-separated list of tracts", type=str, default="5063,4848,4849")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    butler = dafButler.Butler(args.repo)
    skymap = butler.get(skymap.BaseSkyMap.SKYMAP_DATASET_TYPE_NAME, skymap=args.skymap_name, collections=args.skymap_collections)

    tab = Table.read(args.mosaics_table_filepath)
    tracts = [int(x) for x in args.tracts.split(",")]

    make_patch_cutouts(
        mosaic_table=tab,
        filepath=args.image_filepath,
        band=args.band,
        datasettype="euclid_q1_coadd",
        skymap=skymap,
        skymap_name=args.skymap_name,
        save_directory=args.save_directory,
        tracts=tracts,
    )