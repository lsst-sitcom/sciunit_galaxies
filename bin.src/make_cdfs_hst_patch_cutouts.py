import argparse

import lsst.daf.butler as dafButler
import lsst.skymap as skymap
from lsst.sitcom.sciunit.galaxies.cdfs_hst import make_patch_cutouts

if __name__ == '__main__':
    parser = argparse.ArgumentParser("make_patch_cutouts")
    parser.add_argument("--band", help="Name of the filter/band", type=str, default=None)
    parser.add_argument("--datasettype", help="Butler dataset type name", type=str, default=None)
    parser.add_argument("--filename_format", help="Output filename format. ", type=str, default=None)
    parser.add_argument("--image_filepath", help="Filepath of non-LSST image to make patch cutouts from.", type=str)
    parser.add_argument("--repo", help="Butler repo to load skymap from", type=str, default="/repo/main")
    parser.add_argument("--save_directory", help="Directory in which patch cutouts will be saved.", type=str)
    parser.add_argument("--skymap_collections", help="Collections to query for skymap", type=str, default="skymaps")
    parser.add_argument("--skymap_name", help="Name of the skymap", type=str, default="lsst_cells_v1")
    parser.add_argument("--weight_filepath", help="Filepath of weight image, if any", type=str, default=None)
    parser.add_argument("--is_weight_variance", help="Whether the weight image is a variance map (else assumed to be inverse variance)",
                        type=bool, default=False)
    args = parser.parse_args()
    butler = dafButler.Butler(args.repo)
    skymap = butler.get(skymap.BaseSkyMap.SKYMAP_DATASET_TYPE_NAME, skymap=args.skymap_name, collections=args.skymap_collections)
    make_patch_cutouts(
        image_filepath=args.image_filepath,
        skymap=skymap,
        skymap_name=args.skymap_name,
        datasettype=args.datasettype,
        filename_format=args.filename_format,
        save_directory=args.save_directory,
        weight_filepath=args.weight_filepath,
        is_weight_variance=args.is_weight_variance,
        band=args.band,
    )
