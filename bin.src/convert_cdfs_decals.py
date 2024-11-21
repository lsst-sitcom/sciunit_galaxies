import astropy.table as apTab
import astropy.units as u
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
import pyarrow.parquet as pq

# from https://www.legacysurvey.org/viewer/ls-dr10/cat.fits?ralo=52.14077598745257&rahi=54.03427611473381&declo=-28.35063896360024&dechi=-26.684313552983653

name_tab = "decals_dr10_lsst_cells_v1_5063"

tab_ap = apTab.Table.read(f"{name_tab}.fits")

columns = (
    ("RELEASE", "", "Integer denoting the camera and filter set used, which will be unique for a given processing run of the data (RELEASE is documented here)"),
    ("BRICKID", "", "A unique Brick ID (in the range [1, 662174])"),
    ("BRICKNAME", "None",'Name of brick, encoding the brick sky position, eg "1126p222" near RA=112.6, Dec=+22.2'),
    ("OBJID", "", "Catalog object number within this brick; a unique identifier hash is RELEASE,BRICKID,OBJID; OBJID spans [0,N-1] and is contiguously enumerated within each blob"),
    ("TYPE", "None",'Morphological model: "PSF"=stellar, "REX"="round exponential galaxy" = round EXP galaxy with a variable radius, "EXP"=exponential, "DEV"=deVauc, "SER"=Sersic, "DUP"==Gaia source fit by different model. See also the larger description.'),
    ("RA", "deg", "Right ascension at equinox J2000"),
    ("DEC", "deg", "Declination at equinox J2000"),
    ("RA_IVAR", "1/deg**2", "Inverse variance of RA (no cosine term!), excluding astrometric calibration errors"),
    ("DEC_IVAR", "1/deg**2", "Inverse variance of DEC, excluding astrometric calibration errors"),
    ("DCHISQ", "", "Difference in χ**2 between successively more-complex model fits: PSF, REX, DEV, EXP, SER. The difference is versus no source."),
    ("EBV", "mag", "Galactic extinction E(B-V) reddening from SFD98, used to compute MW_TRANSMISSION"),
    ("FLUX_G", "nanomaggy", "Model flux in g"),
    ("FLUX_R", "nanomaggy", "Model flux in r"),
    ("FLUX_I", "nanomaggy", "Model flux in i"),
    ("FLUX_Z", "nanomaggy", "Model flux in z"),
    ("FLUX_W1", "nanomaggy", "WISE model flux in W1 (AB)"),
    ("FLUX_W2", "nanomaggy", "WISE model flux in W2 (AB)"),
    ("FLUX_W3", "nanomaggy", "WISE model flux in W3 (AB)"),
    ("FLUX_W4", "nanomaggy", "WISE model flux in W4 (AB)"),
    ("FLUX_IVAR_G", "1/nanomaggy**2", "Inverse variance of FLUX_G"),
    ("FLUX_IVAR_R", "1/nanomaggy**2", "Inverse variance of FLUX_R"),
    ("FLUX_IVAR_I", "1/nanomaggy**2", "Inverse variance of FLUX_I"),
    ("FLUX_IVAR_Z", "1/nanomaggy**2", "Inverse variance of FLUX_Z"),
    ("FLUX_IVAR_W1", "1/nanomaggy**2", "Inverse variance of FLUX_W1 (AB system)"),
    ("FLUX_IVAR_W2", "1/nanomaggy**2", "Inverse variance of FLUX_W2 (AB)"),
    ("FLUX_IVAR_W3", "1/nanomaggy**2", "Inverse variance of FLUX_W3 (AB)"),
    ("FLUX_IVAR_W4", "1/nanomaggy**2", "Inverse variance of FLUX_W4 (AB)"),
    ("MW_TRANSMISSION_G", "", "Galactic transmission in g filter in linear units [0,1]"),
    ("MW_TRANSMISSION_R", "", "Galactic transmission in r filter in linear units [0,1]"),
    ("MW_TRANSMISSION_R", "", "Galactic transmission in i filter in linear units [0,1]"),
    ("MW_TRANSMISSION_Z", "", "Galactic transmission in z filter in linear units [0,1]"),
    ("MW_TRANSMISSION_W1", "", "Galactic transmission in W1 filter in linear units [0,1]"),
    ("MW_TRANSMISSION_W2", "", "Galactic transmission in W2 filter in linear units [0,1]"),
    ("MW_TRANSMISSION_W3", "", "Galactic transmission in W3 filter in linear units [0,1]"),
    ("MW_TRANSMISSION_W4", "", "Galactic transmission in W4 filter in linear units [0,1]"),
    ("NOBS_G", "", "Number of images that contribute to the central pixel in g : filter for this object (not profile-weighted)"),
    ("NOBS_R", "", "Number of images that contribute to the central pixel in r : filter for this object (not profile-weighted)"),
    ("NOBS_I", "", "Number of images that contribute to the central pixel in i : filter for this object (not profile-weighted)"),
    ("NOBS_Z", "", "Number of images that contribute to the central pixel in z : filter for this object (not profile-weighted)"),
    ("NOBS_W1", "", "Number of images that contribute to the central pixel in W1 : filter for this object (not profile-weighted)"),
    ("NOBS_W2", "", "Number of images that contribute to the central pixel in W2 : filter for this object (not profile-weighted)"),
    ("NOBS_W3", "", "Number of images that contribute to the central pixel in W3 : filter for this object (not profile-weighted)"),
    ("NOBS_W4", "", "Number of images that contribute to the central pixel in W4 : filter for this object (not profile-weighted)"),
    ("RCHISQ_G", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in g"),
    ("RCHISQ_R", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in r"),
    ("RCHISQ_I", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in i"),
    ("RCHISQ_Z", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in z"),
    ("RCHISQ_W1", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in W1"),
    ("RCHISQ_W2", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in W2"),
    ("RCHISQ_W3", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in W3"),
    ("RCHISQ_W4", "", "Profile-weighted χ**2 of model fit normalized by the number of pixels in W4"),
    ("FRACFLUX_G", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in g (typically [0,1])"),
    ("FRACFLUX_R", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in r (typically [0,1])"),
    ("FRACFLUX_I", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in i (typically [0,1])"),
    ("FRACFLUX_Z", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in z (typically [0,1])"),
    ("FRACFLUX_W1", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W1 (typically [0,1])"),
    ("FRACFLUX_W2", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W2 (typically [0,1])"),
    ("FRACFLUX_W3", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W3 (typically [0,1])"),
    ("FRACFLUX_W4", "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W4 (typically [0,1])"),
    ("FRACMASKED_G", "", "Profile-weighted fraction of pixels masked from all observations of this object in g , strictly between [0,1]"),
    ("FRACMASKED_R", "", "Profile-weighted fraction of pixels masked from all observations of this object in r , strictly between [0,1]"),
    ("FRACMASKED_I", "", "Profile-weighted fraction of pixels masked from all observations of this object in i , strictly between [0,1]"),
    ("FRACMASKED_Z", "", "Profile-weighted fraction of pixels masked from all observations of this object in z , strictly between [0,1]"),
    ("FRACIN_G", "", "Fraction of a source's flux within the blob in g, near unity for real sources"),
    ("FRACIN_R", "", "Fraction of a source's flux within the blob in r, near unity for real sources"),
    ("FRACIN_I", "", "Fraction of a source's flux within the blob in i, near unity for real sources"),
    ("FRACIN_Z", "", "Fraction of a source's flux within the blob in z, near unity for real sources"),
    ("ANYMASK_G", "", "Bitwise mask set if the central pixel from any image satisfies each condition in g (see the DR10 bitmasks page)"),
    ("ANYMASK_R", "", "Bitwise mask set if the central pixel from any image satisfies each condition in r (see the DR10 bitmasks page)"),
    ("ANYMASK_I", "", "Bitwise mask set if the central pixel from any image satisfies each condition in i (see the DR10 bitmasks page)"),
    ("ANYMASK_Z", "", "Bitwise mask set if the central pixel from any image satisfies each condition in z (see the DR10 bitmasks page)"),
    ("ALLMASK_G", "", "Bitwise mask set if the central pixel from all images satisfy each condition in g (see the DR10 bitmasks page)"),
    ("ALLMASK_R", "", "Bitwise mask set if the central pixel from all images satisfy each condition in r (see the DR10 bitmasks page)"),
    ("ALLMASK_I", "", "Bitwise mask set if the central pixel from all images satisfy each condition in i (see the DR10 bitmasks page)"),
    ("ALLMASK_Z", "", "Bitwise mask set if the central pixel from all images satisfy each condition in z (see the DR10 bitmasks page)"),
    ("WISEMASK_W1", "", "W1 bitmask as cataloged on the DR10 bitmasks page"),
    ("WISEMASK_W2", "", "W2 bitmask as cataloged on the DR10 bitmasks page"),
    ("PSFSIZE_G", "arcsec", "Weighted average PSF FWHM in the g band"),
    ("PSFSIZE_R", "arcsec", "Weighted average PSF FWHM in the r band"),
    ("PSFSIZE_I", "arcsec", "Weighted average PSF FWHM in the i band"),
    ("PSFSIZE_Z", "arcsec", "Weighted average PSF FWHM in the z band"),
    ("PSFDEPTH_G", "1/nanomaggy**2", "For a 5σ point source detection limit in g, 5/(√PSFDEPTH_G) gives flux in nanomaggies and −2.5[log10(5/(√PSFDEPTH_G))−9] gives corresponding magnitude"),
    ("PSFDEPTH_R", "1/nanomaggy**2", "For a 5σ point source detection limit in r, 5/(√PSFDEPTH_R) gives flux in nanomaggies and −2.5[log10(5/(√PSFDEPTH_R))−9] gives corresponding magnitude"),
    ("PSFDEPTH_I", "1/nanomaggy**2", "For a 5σ point source detection limit in i, 5/(√PSFDEPTH_I) gives flux in nanomaggies and −2.5[log10(5/(√PSFDEPTH_I))−9] gives corresponding magnitude"),
    ("PSFDEPTH_Z", "1/nanomaggy**2", "For a 5σ point source detection limit in z, 5/(√PSFDEPTH_Z) gives flux in nanomaggies and −2.5[log10(5/(√PSFDEPTH_Z))−9] gives corresponding magnitude"),
    ("GALDEPTH_G", "1/nanomaggy**2", 'As for PSFDEPTH_G but for a galaxy (0.45" exp, round) detection sensitivity'),
    ("GALDEPTH_R", "1/nanomaggy**2", 'As for PSFDEPTH_R but for a galaxy (0.45" exp, round) detection sensitivity'),
    ("GALDEPTH_I", "1/nanomaggy**2", 'As for PSFDEPTH_I but for a galaxy (0.45" exp, round) detection sensitivity'),
    ("GALDEPTH_Z", "1/nanomaggy**2", 'As for PSFDEPTH_Z but for a galaxy (0.45" exp, round) detection sensitivity'),
    ("PSFDEPTH_W1", "1/nanomaggy**2", "As for PSFDEPTH_G (and also on the AB system) but for WISE W1"),
    ("PSFDEPTH_W2", "1/nanomaggy**2", "As for PSFDEPTH_G (and also on the AB system) but for WISE W2"),
    ("WISE_COADD_ID", "", "unWISE coadd file name for the center of each object"),
    ("SHAPE_R", "arcsec", "Half-light radius of galaxy model for galaxy type TYPE (>0)"),
    ("SHAPE_R_IVAR", "1/arcsec**2", "Inverse variance of SHAPE_R"),
    ("SHAPE_E1", "", "Ellipticity component 1 of galaxy model for galaxy type TYPE"),
    ("SHAPE_E1_IVAR", "", "Inverse variance of SHAPE_E1"),
    ("SHAPE_E2", "", "Ellipticity component 2 of galaxy model for galaxy type TYPE"),
    ("SHAPE_E2_IVAR", "", "Inverse variance of SHAPE_E2"),
    ("FIBERFLUX_G", "nanomaggy", "Predicted g-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("FIBERFLUX_R", "nanomaggy", "Predicted r-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("FIBERFLUX_I", "nanomaggy", "Predicted i-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("FIBERFLUX_Z", "nanomaggy", "Predicted z-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("FIBERTOTFLUX_G", "nanomaggy", "Predicted g-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("FIBERTOTFLUX_R", "nanomaggy", "Predicted r-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("FIBERTOTFLUX_I", "nanomaggy", "Predicted i-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("FIBERTOTFLUX_Z", "nanomaggy", "Predicted z-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("REF_CAT", "None",'Reference catalog source for this star: "T2" for Tycho-2, "GE" for Gaia EDR3, "L3" for the SGA, empty otherwise'),
    ("REF_ID", "None",'Reference catalog identifier for this star; Tyc1*1,000,000+Tyc2*10+Tyc3 for Tycho-2; "sourceid" for Gaia EDR3 and SGA'),
    ("REF_EPOCH", "yr", "Reference catalog reference epoch (eg, 2016.0 for Gaia EDR3)"),
    ("GAIA_PHOT_G_MEAN_MAG", "mag", "Gaia EDR3 G band magnitude"),
    ("GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR", "", "Gaia EDR3 G band signal-to-noise"),
    ("GAIA_PHOT_BP_MEAN_MAG", "mag", "Gaia EDR3 BP magnitude"),
    ("GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR", "", "Gaia EDR3 BP signal-to-noise"),
    ("GAIA_PHOT_RP_MEAN_MAG", "mag", "Gaia EDR3 RP magnitude"),
    ("GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR", "", "Gaia EDR3 RP signal-to-noise"),
    ("GAIA_ASTROMETRIC_EXCESS_NOISE", "", "Gaia EDR3 astrometric excess noise"),
    ("GAIA_DUPLICATED_SOURCE", "", "Gaia EDR3 duplicated source flag (1/0 for True/False)"),
    ("GAIA_PHOT_BP_RP_EXCESS_FACTOR", "", "Gaia EDR3 BP/RP excess factor"),
    ("GAIA_ASTROMETRIC_SIGMA5D_MAX", "mas", "Gaia EDR3 longest semi-major axis of the 5-d error ellipsoid"),
    ("GAIA_ASTROMETRIC_PARAMS_SOLVED", "", "Which astrometric parameters were estimated for a Gaia EDR3 source"),
    ("PARALLAX", "mas", "Reference catalog parallax"),
    ("PARALLAX_IVAR", "1/mas**2", "Reference catalog inverse-variance on PARALLAX"),
    ("PMRA", "mas/yr", "Reference catalog proper motion in RA direction (μ∗α≡μαcosδ) in the ICRS at REF_EPOCH"),
    ("PMRA_IVAR", "1/(mas/yr)**2", "Reference catalog inverse-variance on PMRA"),
    ("PMDEC", "mas/yr", "Reference catalog proper motion in Dec direction (μδ) in the ICRS at REF_EPOCH"),
    ("PMDEC_IVAR", "1/(mas/yr)**2", "Reference catalog inverse-variance on PMDEC"),
    ("MASKBITS", "", "Bitwise mask indicating that an object touches a pixel in the coadd/*/*/*maskbits* maps (see the DR10 bitmasks page)"),
    ("FITBITS", "", "Bitwise mask detailing properties of how a source was fit (see the DR10 bitmasks page)"),
    ("SERSIC", "", 'Power-law index for the Sersic profile model (TYPE="SER")'),
)

# Identical to (0*u.ABmag).to(u.Jy)
nJy_per_nmgy = (1*u.nanomaggy).to(u.nJy, u.zero_point_flux((0*u.ABmag).to(u.Jy))).value

for values in columns:
    if len(values) != 3:
        print(f"{values[0]} len={len(values)}")
        continue
    name, unit, desc = values
    name_lower = name.lower()
    if name_lower in tab_ap.colnames:
        column = tab_ap[name_lower]
        if unit == "nanomaggy":
            unit = "nJy"
            tab_ap[name_lower] *= nJy_per_nmgy
        column.unit = unit
        column.desc = desc
    else:
        print(f"{name_lower} column not found")

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, f"{name_tab}.parq")
