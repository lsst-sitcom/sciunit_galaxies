import pandas as pd  
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as numpy
import argparse

def make_patch_cutouts(data_filepath, cutout_info_csv_filepath, save_directory):
    # read in image
    with fits.open(data_filepath) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        wcs = WCS(header)

    # read in cutout info
    cutout_info = pd.read_csv(cutout_info_csv_filepath)

    for i, patch in cutout_info.iterrows():
        # make cutout
        center = SkyCoord(patch["center_ra_deg"], patch["center_dec_deg"], unit=u.degree)
        patch_cutout = Cutout2D(data, center, (u.Quantity(patch["height_deg"], unit=u.degree), 
                                                u.Quantity(patch["width_deg"], unit=u.degree)), 
                                                wcs=wcs, copy=True)

        # create new HDU and update header values
        patch_hdu = fits.PrimaryHDU(data=patch_cutout.data, header=header)
        patch_hdu.header.update(patch_cutout.wcs.to_header())
        patch_hdu.header["TRACT_ID"] = int(patch["tract_id"])
        patch_hdu.header["PATCH_ID"] = int(patch["patch_sequential_id"])

        # save to FITS file in specified directory
        cutout_filename = "tract" + str(int(patch["tract_id"])) + "_patch" + str(int(patch["patch_sequential_id"])) + ".fits"
        patch_hdu.writeto(save_directory + cutout_filename) # might need to change this to overwrite=True
        print(f"Tract {int(patch["tract_id"])} patch {int(patch["patch_sequential_id"])} saved to {save_directory + cutout_filename}")

# taking in command-line inputs
parser = argparse.ArgumentParser("make_patch_cutouts")
parser.add_argument("data_filepath", help="Filepath of non-LSST image to make patch cutouts from.", type=str)
parser.add_argument("cutout_info_csv_filepath", help="Filepath of CSV with tract/patch cutout info.", type=str)
parser.add_argument("save_directory", help="Directory in which patch cutouts will be saved.", type=str)
args = parser.parse_args()
make_patch_cutouts(args.data_filepath, args.cutout_info_csv_filepath, args.save_directory)