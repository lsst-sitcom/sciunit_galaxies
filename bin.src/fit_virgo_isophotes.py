# Fit isophotes to derive surface brightness profiles for Virgo galaxies
# This will take forever with main photutils - make sure you install this branch:
# https://github.com/taranu/photutils/tree/cython-build-ellipse-model
# Ideally, install this branch which enables parallel evaluation:
# https://github.com/taranu/photutils/tree/parallel-build-ellipse-model
# like so:
# CFLAGS="${CFLAGS} -fopenmp" LDFLAGS="${LDFLAGS} -lomp" pip install .
# (see https://github.com/astropy/photutils/pull/2046#issuecomment-2923469329)

from copy import deepcopy

import astropy
import lsst.daf.butler as dafButler
from lsst.geom import Box2I, degrees, Extent2D, Point2D, Point2I, SpherePoint
from lsst.multiprofit.plotting.reference_data import bands_weights_lsst
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import photutils
from photutils.isophote import build_ellipse_model, Ellipse, EllipseGeometry, Isophote, IsophoteList
from scipy.stats import mode

mpl.rcParams.update({"image.origin": "lower", "font.size": 13, "figure.figsize": (18, 18)})
# "u/jbosch/DM-50980/M49/075-cb-padded/b012/g008"
butler = dafButler.Butler("/repo/embargo", collections="u/jbosch/DM-50980/M49/filtered/1")
skymap = "discrete_lsstcam_rgb_m49_v1"
dataset = "deep_coadd_lf_subtracted.image"

# If True, subtract the image by the mode outside of the model (if positive)
subtract_mode = True
# If True (and subtract_mode == False), subtract the outermost isophote value
subtract_outer = False
do_m49 = True
rebin = True

if not do_m49:
    obj_name = "ngc4261"
    centroid_sky = SpherePoint(184.8467336, 5.82491667, degrees)
    # In practice, it's hard to keep the outer isophotes behaving well without
    # rebinning over such a wide field
    cutout_shape = (11700, 7800) if rebin else (9000, 6000)
else:
    obj_name = "m49"
    centroid_sky = SpherePoint(187.4449, 8.0004, degrees)
    cutout_shape = (14000, 12000)
    # bin 2x2 pixels
    rebin = True

redo_singleband = False
filename = f"{obj_name}_ellipse_resid_asinh{'_rebin' if rebin else ''}"

skymapInfo = butler.get("skyMap", skymap=skymap, collections="skymaps")
tract = skymapInfo.findTract(centroid_sky).getId()
tractInfo = skymapInfo[tract]
wcs = tractInfo.wcs
centroid = wcs.skyToPixel(centroid_sky)

begin = Point2D(*centroid)
begin.shift(Extent2D(-cutout_shape[1]/2., -cutout_shape[0]/2.))

patchInfo_begin = tractInfo.findPatch(wcs.pixelToSky(begin))

end = Point2D(*centroid)
end.shift(Extent2D(cutout_shape[1]/2., cutout_shape[0]/2.))

patchInfo_end = tractInfo.findPatch(wcs.pixelToSky(end))

x_begin = int(begin[0])
y_begin = int(begin[1])
x_end = x_begin + cutout_shape[1]
y_end = y_begin + cutout_shape[0]

residuals = {}
absmaxes = (4, 6, 8)

bands_rgb = ("i", "r", "g")
bands = (bands_rgb + ("u",))

images_rgb = {}


def isophote_list_to_table(ellipses: IsophoteList) -> astropy.table.Table:
    table = ellipses.to_table()
    n_rows = len(table)
    stop_code, class_name, sample_class_name, valid = ([None]*n_rows for _ in range(4))
    high_order = {k: np.empty(n_rows, dtype=np.float64) for k in ("a3", "a4", "b3", "b4")}
    for idx, ell in enumerate(ellipses):
        stop_code[idx] = ell.stop_code
        class_name[idx] = ell.__class__.__name__
        sample = getattr(ell, "sample")
        sample_class_name[idx] = sample.__class__.__name__ if sample is not None else ""
        valid[idx] = getattr(sample, "sample", True)
        for k, array in high_order.items():
            array[idx] = getattr(ell, k, np.nan)

    table["class_name"] = class_name
    table["sample_class_name"] = sample_class_name
    table["stop_code"] = stop_code
    table["valid"] = valid
    for k, array in high_order.items():
        table[k] = array
    return table


def isophote_table_to_list(table: astropy.table.Table, image: np.ndarray) -> IsophoteList:
    sample_key_map = {
        "eps": "ellipticity",
        "position_angle": "pa",
    }
    isophotes = [None]*len(table)

    for idx, row in enumerate(table):
        kwargs_isophote = {}
        class_name = row["class_name"]
        class_isophote = getattr(photutils.isophote.isophote, class_name)
        if class_name == "Isophote":
            for key in ("niter", "valid", "stop_code"):
                kwargs_isophote[key] = row[key]
        sample_class_name = row["sample_class_name"]
        class_sample = getattr(photutils.isophote.sample, sample_class_name)
        kwargs_sample = {"image": image, "sma": row["sma"], "x0": row["x0"], "y0": row["x0"]}
        if sample_class_name == "EllipseSample":
            for key in ("eps", "position_angle"):
                value = row[sample_key_map.get(key, key)]
                # astropy quantities must be returned to numpy floats
                kwargs_sample[key] = getattr(value, "value", value)
        sample = class_sample(**kwargs_sample)
        sample.extract()
        isophote = class_isophote(sample=sample, **kwargs_isophote)
        isophotes[idx] = isophote
    return IsophoteList(isophotes)


def fit_ellipses(
    image: np.ndarray, maxsma: float | None = None, offset_x0: float = 0., offset_y0: float = 0,
    eps0: float = 0.2, pa0: float = 0, sma0: float | None = None, fix_center: bool = False,
):
    if maxsma is None:
        maxsma = 0.8*np.sqrt(image.size)
    if sma0 is None:
        sma0 = maxsma/10.
    ellipse = Ellipse(
        image,
        geometry=EllipseGeometry(
            x0=image.shape[1] / 2. + offset_x0,
            y0=image.shape[0] / 2. + offset_y0,
            sma=sma, eps=eps0, pa=pa0,
            fix_center=fix_center,
        ),
    )
    iteration = 5
    while iteration >= 0:
        fit = ellipse.fit_image(nclip=5, sma0=sma0, maxsma=maxsma, step=0.1)

        for idx_rev, fitvals in enumerate(reversed(fit)):
            good_fv = all(
                np.isfinite(x) for x in (fitvals.sma, fitvals.intens, fitvals.pa, fitvals.eps)
            )
            good_fv &= (fitvals.stop_code <= 1) or (4 <= fitvals.stop_code <= 5)
            if not good_fv:
                print(
                    f"{iteration=} has bad sma={fitvals.sma}, intens={fitvals.intens}, pa={fitvals.pa},"
                    f" eps={fitvals.eps}, and/or stop_code={fitvals.stop_code} at {idx_rev=} with {maxsma=}"
                )
                if np.isfinite(fitvals.sma):
                    maxsma = 0.98*fitvals.sma
                else:
                    maxsma = 0.95*maxsma
                iteration = 0
                break
        iteration -= 1

    fit_table = fit.to_table()

    fit_pos = []
    fit_neg = []
    for idx, ell in enumerate(fit):
        # Don't trust high order moments for large components
        if ell.sma > sma:
            ell.a3 = 0
            ell.a4 = 0
            ell.b3 = 0
            ell.b4 = 0
            ell.x0_err, ell.y0_err, ell.pa_err, ell.ellip_err, ell.grad = [1e-6]*5
            ell.grad_error = None

        # Besides being unphysical, negative components don't render properly
        (fit_pos if (ell.intens > 0) else fit_neg).append(ell)

    model_neg = None
    if fit_neg:
        try:
            iso_med = deepcopy(fit_pos[-1])
            iso_med.intens = 0
            iso_last_pos = fit_pos[-1]
            iso_first_neg = fit_neg[0]
            weight_neg = iso_last_pos.intens / (iso_last_pos.intens - fit_neg[0].intens)
            weight_pos = 1.0 - weight_neg
            for attr in ("sma", "eps", "pa"):
                value_med = weight_pos * getattr(iso_last_pos, attr) + weight_neg * getattr(iso_first_neg,
                                                                                            attr)
                setattr(iso_med.sample.geometry, attr, value_med)
            fit_pos.append(iso_med)

            for ell in fit_neg:
                ell.intens = -ell.intens

            fit_neg = [iso_med] + fit_neg

            model_neg = build_ellipse_model(
                image.shape, IsophoteList(fit_neg),
                num_threads=16, high_harmonics=False, phi_min=0.05, sma_interval=0.1,
            )
        except ValueError as e:
            print(f"Can't incorporate negative isophotes due to {e=}")
    model = build_ellipse_model(
        image.shape, IsophoteList(fit_pos),
        num_threads=16, high_harmonics=True, phi_min=0.02, sma_interval=0.05,
    )
    if (model_neg is not None) and np.all(np.isfinite(model_neg)):
        model -= model_neg

    return fit, model, fit_table


if do_m49:
    kwargs_fit_ellipses = dict(
        offset_x0 = 0.86 - 0.5*rebin,
        offset_y0 = 1.2 - 0.5*rebin,
        eps0=0.175,
        pa0=1.2,
        sma0 = 200 / (1 + rebin),
        fix_center = True,
    )
else:
    kwargs_fit_ellipses = dict(
        offset_x0=0.59 - 0.5*rebin,
        offset_y0=0.56 - 0.5*rebin,
        eps0=0.23,
        pa0=1.2,
        sma0=200 / (1 + rebin),
        fix_center = True,
    )


for band in bands:
    image = np.empty(cutout_shape, dtype=float)

    for patch_y in range(patchInfo_begin.index.y, patchInfo_end.index.y + 1):
        for patch_x in range(patchInfo_begin.index.x, patchInfo_end.index.x + 1):
            patch = tractInfo.getSequentialPatchIndexFromPair((patch_x, patch_y))
            patchInfo = tractInfo[patch]
            patch_bbox = patchInfo.inner_bbox

            is_end_y = (patch_y == patchInfo_end.index.y)
            is_end_x = (patch_x == patchInfo_end.index.x)

            x_begin_patch = max(patch_bbox.getBeginX(), x_begin)
            x_end_patch = min(patch_bbox.getEndX(), x_end - is_end_x)
            y_begin_patch = max(patch_bbox.getBeginY(), y_begin)
            y_end_patch = min(patch_bbox.getEndY(), y_end - is_end_y)

            bbox_read = Box2I(
                minimum=Point2I(x_begin_patch, y_begin_patch),
                maximum=Point2I(x_end_patch, y_end_patch),
            )

            patch_array = butler.get(
                dataset, skymap=skymap, tract=tract, patch=patch, band=band, parameters={"bbox": bbox_read}
            ).array

            image[
                y_begin_patch - y_begin:y_end_patch - y_begin + 1,
                x_begin_patch - x_begin:x_end_patch - x_begin + 1,
            ] = patch_array

    sma = 500
    if rebin:
        sma *= 0.5
        image = (
            image[::2, ::2] + image[1::2,::2] + image[::2,1::2] + image[1::2,1::2]
        )/4.

    if redo_singleband:
        fit, model, fit_table = fit_ellipses(image, **kwargs_fit_ellipses)
        fit_table.write(f"{filename}_ellipses_{band}.ecsv", overwrite=True)
        bglevel = max(
            mode(np.round((image*1000.).flat)/1000.) if subtract_mode else (
                fit[-1].intens if subtract_outer else 0.
            ),
            0.,
        )
        diff = np.arcsinh((image - model - bglevel*(model == 0))*np.isfinite(model))
        residuals[band] = diff

        for absmax in absmaxes:
            plt.imsave(
                f"{filename}_pm{absmax}_{band}.png", np.clip(diff, -absmax, absmax), cmap="gray",
            )
    else:
        if band in bands_rgb:
            images_rgb[band] = image

if len(bands_rgb) == 3:
    weight_mid = bands_weights_lsst[bands_rgb[1]]
    if not redo_singleband:
        if do_m49:
            maxsma_init, maxsma_redo = 6000 / (1 + rebin), 6600 / (1 + rebin)
        else:
            maxsma_init, maxsma_redo = 3800 / (1 + rebin), 4200 / (1 + rebin)

        image = np.sum(list(images_rgb.values()), axis=0)
        fit, model, fit_table = fit_ellipses(image, maxsma=maxsma_init, **kwargs_fit_ellipses)
        # fits are annoyingly sensitive to this parameter
        # either they make ellipses larger than the image which don't render
        # well (because of the stopping condition at +/- 0 relative to P.A.)
        # ... or the outer ellipse position angles diverge enough that the
        # ellipses start intersecting each other
        maxsma = maxsma_redo
        fits_all = [fit]

        for idx in range(2):
            image_new = image.copy()
            replace = (np.abs((image - model) / (np.abs(image) + 1e-10)) > 0.8) & (
                (model > 5) | ((model > 2) & (image > 3))
            )
            print(f"{idx=} replacing {np.sum(replace)}/{np.sum(model>0)} pixels with model")
            image_new[replace] = model[replace]
            fit, model, fit_table = fit_ellipses(image_new, maxsma=maxsma, **kwargs_fit_ellipses)
            if idx == 0:
                maxsma += 300 / (1 + rebin)
                if not do_m49:
                    maxsma += 500 / (1 + rebin)
            fits_all.append(fit)

        fit_table.write(f"{filename}_ellipses_{','.join(bands_rgb)}.ecsv", overwrite=True)

        for band in bands_rgb:
            image_band = images_rgb[band]
            for ell in fit:
                ell.sample.image = image_band
                ell.sample.values = None
                ell.sample.update()
                if ell.__class__.__name__ == "CentralPixel":
                    ell.__init__(ell.sample)
                else:
                    ell.__init__(ell.sample, ell.niter, ell.valid, ell.stop_code)
                # Don't trust high order moments for large components
                if ell.sma > sma:
                    ell.a3 = 0
                    ell.a4 = 0
                    ell.b3 = 0
                    ell.b4 = 0
                    ell.x0_err, ell.y0_err, ell.pa_err, ell.ellip_err, ell.grad = [1e-6] * 5
                    ell.grad_error = None

            fit.to_table().write(f"{filename}_ellipses_{band}.ecsv", overwrite=True)

            model = build_ellipse_model(
                image_band.shape, fit,
                num_threads=16, high_harmonics=True, phi_min=0.02, sma_interval=0.05,
            )
            bglevel = max(
                mode(np.round((image_band * 1000.).flat) / 1000.).mode if subtract_mode else (
                    fit[-1].intens if subtract_outer else 0.
                ),
                0.,
            )
            diff = np.arcsinh((image_band - model - bglevel*(model == 0)) * np.isfinite(model))
            residuals[band] = diff
            for absmax in absmaxes:
                plt.imsave(
                    f"{filename}_pm{absmax}_{band}.png", np.clip(diff, -absmax, absmax), cmap="gray",
                )

    rgb = np.stack([
        np.arcsinh(np.sinh(residuals[band])*bands_weights_lsst[band]/weight_mid)
        for band in bands_rgb
    ], axis=2)
    name_rgb = ','.join(bands_rgb)
    for absmax in absmaxes:
        plt.imsave(
            f"{filename}_pm{absmax}_{name_rgb}.png",
            np.clip((absmax + rgb)/(2*absmax), 0., 1),
        )
