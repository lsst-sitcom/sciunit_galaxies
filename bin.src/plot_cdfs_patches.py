from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from collections import defaultdict
import gc
import lsst.daf.butler as dafButler
from lsst.geom import SpherePoint, degrees
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce

mpl.rcParams.update({"image.origin": "lower", 'font.size': 16})

do_rgb = False
path_cdfs = "/sdf/data/rubin/user/dtaranu/tickets/cdfs/"

if do_rgb:
    from astropy.visualization import make_lupton_rgb
    bands = ("F814W", "F606W", "F435W")
else:
    bands = ("F775W",)

wcses = {}
imgs_down = {}

for band in bands:
    fits_band = fits.open(f"{path_cdfs}hlsp_hlf_hst_acs-30mas_goodss_{band.lower()}_v2.0_sci.fits.gz")
    img = fits_band[0].data
    imgs_down[band] = block_reduce(img, block_size=(25, 25), func=np.mean)*(
        10 ** (((u.nJy).to(u.ABmag) - fits_band[0].header["ZEROPNT"])/2.5)
    )
    wcses[band] = WCS(fits_band[0])
    del fits_band
    gc.collect()

wcs_hst = wcses[bands[0]]

if do_rgb:
    import astropy.units as u
    abs_mag_hst = {
        "F435W": 5.35,
        "F606W": 4.72,
        "F775W": 4.52,
        "F814W": 4.52,
    }
    weights_hst = 1/(u.ABmag.to(u.Jy, [abs_mag_hst[band] for band in bands]))
    weights_hst_mean = np.mean(weights_hst)
    bands_weights_hst = {
        band: weight_hst/weights_hst_mean for band, weight_hst in zip(bands, weights_hst)
    }
    img_down = make_lupton_rgb(
        *(img_down*bands_weights_hst[band] for band, img_down in imgs_down.items()),
        Q=8,
        stretch=1.0,
    )
else:
    img_down = imgs_down[bands[0]]
    no_data = img_down == 0
    img_down[no_data] = np.nan
    img_down[~no_data] = np.log10(np.clip(img_down[~no_data], 0, 0.9) + 0.01)


butler = dafButler.Butler("/repo/embargo")
skymap = butler.get("skyMap", skymap="lsst_cells_v1", collections="skymaps")

radec_begin, radec_end = (
    wcs_hst.pixel_to_world(xy[0], xy[1]) for xy in ((0, 0), (img.shape[1], img.shape[0]))
)
(ra_min, dec_min), (ra_max, dec_max) = (
    (radec.ra.value, radec.dec.value) for radec in (radec_begin, radec_end)
)
ra_lo, ra_hi = (ra_min, ra_max) if (ra_max > ra_min) else (ra_max, ra_min)
dec_lo, dec_hi = (dec_min, dec_max) if (dec_max > dec_min) else (dec_max, dec_min)

tracts = [
    skymap.findTract(SpherePoint(radec[0], radec[1], degrees))
    for radec in ((ra_min, dec_min), (ra_min, dec_max), (ra_max, dec_min), (ra_max, dec_max))
]
tracts_counts = defaultdict(int)
for tract in tracts:
    tracts_counts[tract] += 1

kwargs_imshow = {}
if do_rgb:
    color_patch = "gray"
else:
    color_patch = "red"
    cmap = mpl.colormaps.get_cmap("gray")
    cmap.set_bad(color="darkblue")
    kwargs_imshow["cmap"] = cmap
plt.autoscale(False)

fig, ax = plt.subplots(figsize=(24, 24))
ax.imshow(
    img_down, extent=[ra_min, ra_max, dec_min, dec_max], **kwargs_imshow
)

for tract in tracts_counts.keys():
    for patch in tract:
        corners = patch.getInnerBBox().getCorners()
        radecs = np.zeros((5, 2))
        corner = corners[-1]

        within = False
        for idx, corner in enumerate(corners):
            ra, dec = (x.asDegrees() for x in tract.wcs.pixelToSky(corner[0], corner[1]))
            if (ra >= ra_lo) and (ra <= ra_hi) and (dec >= dec_lo) and (dec <= dec_hi):
                within = True
            radecs[idx, :] = [np.clip(ra, ra_lo, ra_hi), np.clip(dec, dec_lo, dec_hi)]
        if within:
            radecs[-1, :] = radecs[0, :]
            ax.plot(radecs[:, 0], radecs[:, 1], c=color_patch)
            ax.text(
                (radecs[0, 0] + radecs[1, 0])/2.,
                (radecs[0, 1] + radecs[2, 1])/2.,
                str(patch.sequential_index),
                c=color_patch,
            )

plt.show()
