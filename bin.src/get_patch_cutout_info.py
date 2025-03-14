import numpy as np
import lsst.daf.butler as daf_butler
import lsst.geom as geom
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import argparse

# get the coordinates of the bounding box of the non-LSST image
def wcs_four_corners_spherepoint(filepath):
    with fits.open(filepath) as hdu:
        header = hdu[0].header
        data_dimensions = hdu[0].data.shape
        wcs = WCS(header)

    x_dim = data_dimensions[1]
    y_dim = data_dimensions[0]
    
    top_left = [0, 0]
    top_right = [x_dim - 1, 0]
    bottom_left = [0, y_dim - 1]
    bottom_right = [x_dim - 1, y_dim - 1]
    
    # last argument is for zero-based indices instead of the default 1 for FITS
    top_left = wcs.wcs_pix2world(top_left[0], top_left[1], 0)
    top_right = wcs.wcs_pix2world(top_right[0], top_right[1], 0) 
    bottom_left = wcs.wcs_pix2world(bottom_left[0], bottom_left[1], 0) 
    bottom_right = wcs.wcs_pix2world(bottom_right[0], bottom_right[1], 0) 
    
    return [geom.SpherePoint(top_left[0], top_left[1], geom.degrees),
           geom.SpherePoint(top_right[0], top_right[1], geom.degrees),
           geom.SpherePoint(bottom_left[0], bottom_left[1], geom.degrees),
           geom.SpherePoint(bottom_right[0], bottom_right[1], geom.degrees)]


# returns a list of tracts, which have associated lists of patches
def get_unique_tract_patch_overlap_list(four_corners):
    repo = '/sdf/group/rubin/repo/main/butler.yaml'
    collections = ['LSSTCam/raw/all']
    butler = daf_butler.Butler(repo, collections=collections)
    name_skymap = "lsst_cells_v1"
    skymap = butler.get("skyMap", skymap=name_skymap, collections="skymaps")

    # Based on some tests, it seems that this function doesn't return patch duplicates (by default).
    tract_patch_list = skymap.findTractPatchList(four_corners)

    # The first index is tract family
    # The second index is to choose the tract info (0) or patch list (1)
    # The third index (inside the patch list) is individual patches
    # Tract IDs can be accessed with .tract_id, patch IDs can be accessed with .sequential_index
    return tract_patch_list # TODO EVENTUALLY: can I return this in a more user-friendly way? maybe a pandas df with tract_id and
    # patch sequential index, and then a column for just the tractInfo object so that I don't have to unpack that fully?


# creates a pandas dataframe with the info needed to make patch cutouts (center_ra, center_dec, height, and width, all in degrees)
# and also tract and patch IDs
def get_patch_cutout_info(tracts_patches_list):
    tract_id_list = []
    patch_id_list = []
    center_ra_deg_list = []
    center_dec_deg_list = []
    height_deg_list = []
    width_deg_list = []
    
    for tract_family in tracts_patches_list:
        # THIS CODE ADAPTED FROM https://github.com/lsst-sitcom/sciunit_galaxies/blob/main/bin.src/compare_cdfs.py
        patchList = tract_family[1]
        tractInfo = tract_family[0]
        wcs = tractInfo.wcs
        for patchInfo in patchList:
            bbox_outer = patchInfo.outer_bbox

            # The asterisks here unpack the tuple output into two separate arguments
            (ra_begin, dec_begin), (ra_end, dec_end) = (
                (x.getRa().asDegrees(), x.getDec().asDegrees())
                for x in (
                    wcs.pixelToSky(*(bbox_outer.getBegin())),
                    wcs.pixelToSky(*(bbox_outer.getEnd())),
                )
            )

            center_ra = (ra_end + ra_begin) / 2.
            center_dec = (dec_end + dec_begin) / 2.
            height_deg = np.abs(dec_begin - dec_end)
            width_deg = np.abs(ra_begin - ra_end)

            tract_id_list.append(tractInfo.getId())
            patch_id_list.append(patchInfo.getSequentialIndex())
            center_ra_deg_list.append(center_ra)
            center_dec_deg_list.append(center_dec)
            height_deg_list.append(height_deg)
            width_deg_list.append(width_deg)

    all_cutout_info = pd.DataFrame(data = {"tract_id":tract_id_list, "patch_sequential_id":patch_id_list,
                                           "center_ra_deg":center_ra_deg_list, "center_dec_deg":center_dec_deg_list, 
                                           "height_deg":height_deg_list, "width_deg":width_deg_list})
    return all_cutout_info
    

# master function that goes from filepath to CSV patch cutout info
def file_to_patch_cutout_info(filepath, return_csv_name):
    tracts_patches_list = get_unique_tract_patch_overlap_list(wcs_four_corners_spherepoint(filepath))
    cutout_info = get_patch_cutout_info(tracts_patches_list)
    cutout_info.to_csv(return_csv_name, index=False)

    return cutout_info


##################################################################################################################
# running from command-line inputs
parser = argparse.ArgumentParser("file_to_patch_cutout_info")
parser.add_argument("filepath", help="Filepath of non-LSST image to get tract/patch cutout info on.", type=str)
parser.add_argument("return_csv_name", help="Filepath of CSV output.", type=str)
args = parser.parse_args()
cutout_info = file_to_patch_cutout_info(args.filepath, args.return_csv_name)