import astropy.table as apTab
import astropy.units as u
import lsst.daf.butler as dafButler
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size
from lsst.geom import degrees, SpherePoint
import numpy as np
import pyarrow.parquet as pq

# from https://www.legacysurvey.org/viewer/ls-dr10/cat.fits?ralo=52.14077598745257&rahi=54.03427611473381&declo=-28.35063896360024&dechi=-26.684313552983653

skymap = "lsst_cells_v1"
tract = 5063
butler = dafButler.Butler("/repo/main", collections="skymaps")
tractInfo = butler.get("skyMap", skymap=skymap)[tract]

name_tab = f"decals_dr10_{skymap}_{tract}"
tab_ap = apTab.Table.read(f"{name_tab}.fits")

columns = (
    ("release", np.int16, 1, "", "Integer denoting the camera and filter set used, which will be unique for a given processing run of the data (url='../../release' (as documented here))"),
    ("brickid", np.int32, 1, "", "Brick ID [1,662174]"),
    ("brickname", "S8", 1, "", "Name of brick, encoding the brick sky position, eg '1126p222' near RA=112.6, Dec=+22.2"),
    ("objid", np.int32, 1, "", "Catalog object number within this brick; a unique identifier hash is 'release,brickid,objid';  'objid' spans [0,N-1] and is contiguously enumerated within each brick"),
    ("brick_primary", bool, 1, "", "True if the object is within the brick boundary"),
    ("maskbits", np.int32, 1, "", "Bitwise mask indicating that an object touches a pixel in the maskbits maps"),
    ("fitbits", np.int16, 1, "", "Bitwise mask detailing pecularities of how an object was fit, as cataloged on the DR10 bitmasks page"),
    ("type", "S3", 1, "", "Morphological model: 'PSF'=stellar, 'REX'='round exponential galaxy', 'DEV'=deVauc, 'EXP'=exponential, 'SER'=Sersic, 'DUP'=Gaia source fit by different model."),
    ("ra", np.float64, 1, "deg", "Right ascension at equinox J2000"),
    ("dec", np.float64, 1, "deg", "Declination at equinox J2000"),
    ("ra_ivar", np.float32, 1, "1/deg²", "Inverse variance of RA (no cosine term!), excluding astrometric calibration errors"),
    ("dec_ivar", np.float32, 1, "1/deg²", "Inverse variance of DEC, excluding astrometric calibration errors"),
    ("bx", np.float32, 1, "pix", "X position (0-indexed) of coordinates in the brick image stack (<em>i.e.</em> in the <em>e.g.</em> legacysurvey-<brick>-image-g.fits.fz coadd file)"),
    ("by", np.float32, 1, "pix", "Y position (0-indexed) of coordinates in brick image stack"),
    ("dchisq", np.float32, 5, "", "Difference in χ² between successively more-complex model fits: PSF, REX, DEV, EXP, SER.  The difference is versus no source."),
    ("ebv", np.float32, 1, "mag", "Galactic extinction E(B-V) reddening from url='https://ui.adsabs.harvard.edu/abs/1998ApJ...500..525S/abstract' (SFD98') used to compute the 'mw_transmission_' columns"),
    ("mjd_min", np.float64, 1, "days", "Minimum Modified Julian Date of observations used to construct the model of this object"),
    ("mjd_max", np.float64, 1, "days", "Maximum Modified Julian Date of observations used to construct the model of this object"),
    ("ref_cat", "S2", 1, "", "Reference catalog source for this star: 'T2' for url='https://heasarc.gsfc.nasa.gov/W3Browse/all/tycho2.html' (Tycho-2') 'GE' for url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3') 'L3' for the url='../../sga/sga2020' (SGA') empty otherwise"),
    ("ref_id", np.int64, 1, "", "Reference catalog identifier for this star; Tyc1*1,000,000+Tyc2*10+Tyc3 for Tycho2; 'sourceid' for url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) and url='https://www.legacysurvey.org/sga/sga2020/' (SGA)"),
    ("pmra", np.float32, 1, "mas/yr", "Reference catalog proper motion in RA direction (mu_alpha^* ≡ mu_alpha*cos(delta)) in the ICRS at 'ref_epoch'"),
    ("pmdec", np.float32, 1, "mas/yr", "Reference catalog proper motion in Dec direction (mu_delta) in the ICRS at 'ref_epoch'"),
    ("parallax", np.float32, 1, "mas", "Reference catalog parallax"),
    ("pmra_ivar", np.float32, 1, "1/(mas/yr)²", "Reference catalog inverse-variance on 'pmra'"),
    ("pmdec_ivar", np.float32, 1, "1/(mas/yr)²", "Reference catalog inverse-variance on 'pmdec'"),
    ("parallax_ivar", np.float32, 1, "1/mas²", "Reference catalog inverse-variance on 'parallax'"),
    ("ref_epoch", np.float32, 1, "yr", "Reference catalog reference epoch (eg, 2015.5 for url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3))"),
    ("gaia_phot_g_mean_mag", np.float32, 1, "mag", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) G band mag"),
    ("gaia_phot_g_mean_flux_over_error", np.float32, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) G band signal-to-noise"),
    ("gaia_phot_g_n_obs", np.dtype("<i4"), 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) G band number of observations"),
    ("gaia_phot_bp_mean_mag", np.float32, 1, "mag", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) BP mag"),
    ("gaia_phot_bp_mean_flux_over_error", np.float32, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) BP signal-to-noise"),
    ("gaia_phot_bp_n_obs", np.int16, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) BP number of observations"),
    ("gaia_phot_rp_mean_mag", np.float32, 1, "mag", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) RP mag"),
    ("gaia_phot_rp_mean_flux_over_error", np.float32, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) RP signal-to-noise"),
    ("gaia_phot_rp_n_obs", np.int16, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) RP number of observations"),
    ("gaia_phot_variable_flag", bool, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) photometric variable flag"),
    ("gaia_astrometric_excess_noise", np.float32, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) astrometric excess noise"),
    ("gaia_astrometric_excess_noise_sig", np.float32, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) astrometric excess noise uncertainty"),
    ("gaia_astrometric_n_obs_al", np.int16, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) number of astrometric observations along scan direction"),
    ("gaia_astrometric_n_good_obs_al", np.int16, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) number of good astrometric observations along scan direction"),
    ("gaia_astrometric_weight_al", np.float32, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) astrometric weight along scan direction"),
    ("gaia_duplicated_source", bool, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) duplicated source flag"),
    ("gaia_a_g_val", np.float32, 1, "magnitudes", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) line-of-sight extinction in the G band"),
    ("gaia_e_bp_min_rp_val", np.float32, 1, "magnitudes", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) line-of-sight reddening E(BP-RP)"),
    ("gaia_phot_bp_rp_excess_factor", np.float32, 1, "", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) BP/RP excess factor"),
    ("gaia_astrometric_sigma5d_max", np.float32, 1, "mas", "url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) longest semi-major axis of the 5-d error ellipsoid"),
    ("gaia_astrometric_params_solved", np.uint8, 1, "", "Which astrometric parameters were estimated for a url='https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html' (Gaia EDR3) source"),
    ("flux_g", np.float32, 1, "nanomaggy", "model flux in g"),
    ("flux_r", np.float32, 1, "nanomaggy", "model flux in r"),
    ("flux_i", np.float32, 1, "nanomaggy", "model flux in i"),
    ("flux_z", np.float32, 1, "nanomaggy", "model flux in z"),
    ("flux_w1", np.float32, 1, "nanomaggy", "WISE model flux in W1 (AB system)"),
    ("flux_w2", np.float32, 1, "nanomaggy", "WISE model flux in W2 (AB)"),
    ("flux_w3", np.float32, 1, "nanomaggy", "WISE model flux in W3 (AB)"),
    ("flux_w4", np.float32, 1, "nanomaggy", "WISE model flux in W4 (AB)"),
    ("flux_ivar_g", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_g'"),
    ("flux_ivar_r", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_r'"),
    ("flux_ivar_i", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_i'"),
    ("flux_ivar_z", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_z'"),
    ("flux_ivar_w1", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_w1' (AB system)"),
    ("flux_ivar_w2", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_w2' (AB)"),
    ("flux_ivar_w3", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_w3' (AB)"),
    ("flux_ivar_w4", np.float32, 1, "1/nanomaggy²", "Inverse variance of 'flux_w4' (AB)"),
    ("fiberflux_g", np.float32, 1, "nanomaggy", "Predicted g-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("fiberflux_r", np.float32, 1, "nanomaggy", "Predicted r-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("fiberflux_i", np.float32, 1, "nanomaggy", "Predicted i-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("fiberflux_z", np.float32, 1, "nanomaggy", "Predicted z-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing"),
    ("fibertotflux_g", np.float32, 1, "nanomaggy", "Predicted g-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("fibertotflux_r", np.float32, 1, "nanomaggy", "Predicted r-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("fibertotflux_i", np.float32, 1, "nanomaggy", "Predicted i-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("fibertotflux_z", np.float32, 1, "nanomaggy", "Predicted z-band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing"),
    ("apflux_g", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.0] arcsec in g, masked by invvar=0 (inverse variance of zero"),
    ("apflux_r", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.0] arcsec in r, masked by invvar=0"),
    ("apflux_i", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.0] arcsec in i, masked by invvar=0"),
    ("apflux_z", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.0] arcsec in z, masked by invvar=0"),
    ("apflux_resid_g", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added residual images in g, masked by invvar=0"),
    ("apflux_resid_r", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added residual images in r, masked by invvar=0"),
    ("apflux_resid_i", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added residual images in i, masked by invvar=0"),
    ("apflux_resid_z", np.float32, 8, "nanomaggy", "Aperture fluxes on the co-added residual images in z, masked by invvar=0"),
    ("apflux_blobresid_g", np.float32, 8, "nanomaggy", "Aperture fluxes on image-blobmodel residual maps in g, masked by invvar=0"),
    ("apflux_blobresid_r", np.float32, 8, "nanomaggy", "Aperture fluxes on image-blobmodel residual maps in r, masked by invvar=0"),
    ("apflux_blobresid_i", np.float32, 8, "nanomaggy", "Aperture fluxes on image-blobmodel residual maps in i, masked by invvar=0"),
    ("apflux_blobresid_z", np.float32, 8, "nanomaggy", "Aperture fluxes on image-blobmodel residual maps in z, masked by invvar=0"),
    ("apflux_ivar_g", np.float32, 8, "1/nanomaggy²", "Inverse variance of 'apflux_resid_g', masked by invvar=0"),
    ("apflux_ivar_r", np.float32, 8, "1/nanomaggy²", "Inverse variance of 'apflux_resid_r', masked by invvar=0"),
    ("apflux_ivar_i", np.float32, 8, "1/nanomaggy²", "Inverse variance of 'apflux_resid_i', masked by invvar=0"),
    ("apflux_ivar_z", np.float32, 8, "1/nanomaggy²", "Inverse variance of 'apflux_resid_z', masked by invvar=0"),
    ("apflux_masked_g", np.float32, 8, "", "Fraction of pixels masked in g-band aperture flux measurements; 1 means fully masked (ie, fully ignored; contributing zero to the measurement)"),
    ("apflux_masked_r", np.float32, 8, "", "Fraction of pixels masked in r-band aperture flux measurements; 1 means fully masked (ie, fully ignored; contributing zero to the measurement)"),
    ("apflux_masked_i", np.float32, 8, "", "Fraction of pixels masked in i-band aperture flux measurements; 1 means fully masked (ie, fully ignored; contributing zero to the measurement)"),
    ("apflux_masked_z", np.float32, 8, "", "Fraction of pixels masked in z-band aperture flux measurements; 1 means fully masked (ie, fully ignored; contributing zero to the measurement)"),
    ("apflux_w1", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [3, 5, 7, 9, 11]  arcsec in W1, masked by invvar=0"),
    ("apflux_w2", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [3, 5, 7, 9, 11] arcsec in W2, masked by invvar=0"),
    ("apflux_w3", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [3, 5, 7, 9, 11] arcsec in W3, masked by invvar=0"),
    ("apflux_w4", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added images in apertures of radius [3, 5, 7, 9, 11] arcsec in W4, masked by invvar=0"),
    ("apflux_resid_w1", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added residual images in W1, masked by invvar=0"),
    ("apflux_resid_w2", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added residual images in W2, masked by invvar=0"),
    ("apflux_resid_w3", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added residual images in W3, masked by invvar=0"),
    ("apflux_resid_w4", np.float32, 5, "nanomaggy", "Aperture fluxes on the co-added residual images in W4, masked by invvar=0"),
    ("apflux_ivar_w1", np.float32, 5, "1/nanomaggy²", "Inverse variance of 'apflux_resid_w1', masked by invvar=0"),
    ("apflux_ivar_w2", np.float32, 5, "1/nanomaggy²", "Inverse variance of 'apflux_resid_w2', masked by invvar=0"),
    ("apflux_ivar_w3", np.float32, 5, "1/nanomaggy²", "Inverse variance of 'apflux_resid_w3', masked by invvar=0"),
    ("apflux_ivar_w4", np.float32, 5, "1/nanomaggy²", "Inverse variance of 'apflux_resid_w4', masked by invvar=0"),
    ("mw_transmission_g", np.float32, 1, "", "Galactic transmission in g filter in linear units [0, 1]"),
    ("mw_transmission_r", np.float32, 1, "", "Galactic transmission in r filter in linear units [0, 1]"),
    ("mw_transmission_i", np.float32, 1, "", "Galactic transmission in i filter in linear units [0, 1]"),
    ("mw_transmission_z", np.float32, 1, "", "Galactic transmission in z filter in linear units [0, 1]"),
    ("mw_transmission_w1", np.float32, 1, "", "Galactic transmission in W1 filter in linear units [0, 1]"),
    ("mw_transmission_w2", np.float32, 1, "", "Galactic transmission in W2 filter in linear units [0, 1]"),
    ("mw_transmission_w3", np.float32, 1, "", "Galactic transmission in W3 filter in linear units [0, 1]"),
    ("mw_transmission_w4", np.float32, 1, "", "Galactic transmission in W4 filter in linear units [0, 1]"),
    ("nobs_g", np.int16, 1, "", "Number of images that contribute to the central pixel in g filter for this object (not profile-weighted)"),
    ("nobs_r", np.int16, 1, "", "Number of images that contribute to the central pixel in r filter for this object (not profile-weighted)"),
    ("nobs_i", np.int16, 1, "", "Number of images that contribute to the central pixel in i filter for this object (not profile-weighted)"),
    ("nobs_z", np.int16, 1, "", "Number of images that contribute to the central pixel in z filter for this object (not profile-weighted)"),
    ("nobs_w1", np.int16, 1, "", "Number of images that contribute to the central pixel in W1 filter for this object (not profile-weighted)"),
    ("nobs_w2", np.int16, 1, "", "Number of images that contribute to the central pixel in W2 filter for this object (not profile-weighted)"),
    ("nobs_w3", np.int16, 1, "", "Number of images that contribute to the central pixel in W3 filter for this object (not profile-weighted)"),
    ("nobs_w4", np.int16, 1, "", "Number of images that contribute to the central pixel in W4 filter for this object (not profile-weighted)"),
    ("rchisq_g", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in g"),
    ("rchisq_r", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in r"),
    ("rchisq_i", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in i"),
    ("rchisq_z", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in z"),
    ("rchisq_w1", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in W1"),
    ("rchisq_w2", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in W2"),
    ("rchisq_w3", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in W3"),
    ("rchisq_w4", np.float32, 1, "", "Profile-weighted χ² of model fit normalized by the number of pixels in W4"),
    ("fracflux_g", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in g (typically [0,1])"),
    ("fracflux_r", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in r (typically [0,1])"),
    ("fracflux_i", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in i (typically [0,1])"),
    ("fracflux_z", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in z (typically [0,1])"),
    ("fracflux_w1", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W1 (typically [0,1])"),
    ("fracflux_w2", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W2 (typically [0,1])"),
    ("fracflux_w3", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W3 (typically [0,1])"),
    ("fracflux_w4", np.float32, 1, "", "Profile-weighted fraction of the flux from other sources divided by the total flux in W4 (typically [0,1])"),
    ("fracmasked_g", np.float32, 1, "", "Profile-weighted fraction of pixels masked from all observations of this object in g, strictly between [0,1]"),
    ("fracmasked_r", np.float32, 1, "", "Profile-weighted fraction of pixels masked from all observations of this object in r, strictly between [0,1]"),
    ("fracmasked_i", np.float32, 1, "", "Profile-weighted fraction of pixels masked from all observations of this object in i, strictly between [0,1]"),
    ("fracmasked_z", np.float32, 1, "", "Profile-weighted fraction of pixels masked from all observations of this object in z, strictly between [0,1]"),
    ("fracin_g", np.float32, 1, "", "Fraction of a source's flux within the blob in g, near unity for real sources"),
    ("fracin_r", np.float32, 1, "", "Fraction of a source's flux within the blob in r, near unity for real sources"),
    ("fracin_i", np.float32, 1, "", "Fraction of a source's flux within the blob in i, near unity for real sources"),
    ("fracin_z", np.float32, 1, "", "Fraction of a source's flux within the blob in z, near unity for real sources"),
    ("ngood_g", np.int16, 1, "", "Number of <cite>good</cite> (unmasked) images that contribute in g (this quantity is consistent with the <cite>nexp</cite> maps in the url='../files/#image-stacks-south-coadd' (image stacks))"),
    ("ngood_r", np.int16, 1, "", "Number of <cite>good</cite> (unmasked) images that contribute in r (this quantity is consistent with the <cite>nexp</cite> maps in the url='../files/#image-stacks-south-coadd' (image stacks))"),
    ("ngood_i", np.int16, 1, "", "Number of <cite>good</cite> (unmasked) images that contribute in i (this quantity is consistent with the <cite>nexp</cite> maps in the url='../files/#image-stacks-south-coadd' (image stacks))"),
    ("ngood_z", np.int16, 1, "", "Number of <cite>good</cite> (unmasked) images that contribute in z (this quantity is consistent with the <cite>nexp</cite> maps in the url='../files/#image-stacks-south-coadd' (image stacks))"),
    ("anymask_g", np.int16, 1, "", "Bitwise mask set if the central pixel from any image satisfies each condition in g as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("anymask_r", np.int16, 1, "", "Bitwise mask set if the central pixel from any image satisfies each condition in r as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("anymask_i", np.int16, 1, "", "Bitwise mask set if the central pixel from any image satisfies each condition in i as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("anymask_z", np.int16, 1, "", "Bitwise mask set if the central pixel from any image satisfies each condition in z as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("allmask_g", np.int16, 1, "", "Bitwise mask set if the central pixel from all images satisfy each condition in g as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("allmask_r", np.int16, 1, "", "Bitwise mask set if the central pixel from all images satisfy each condition in r as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("allmask_i", np.int16, 1, "", "Bitwise mask set if the central pixel from all images satisfy each condition in i as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("allmask_z", np.int16, 1, "", "Bitwise mask set if the central pixel from all images satisfy each condition in z as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("wisemask_w1", np.uint8, 1, "", "W1 bitmask as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("wisemask_w2", np.uint8, 1, "", "W2 bitmask as cataloged on the url='../bitmasks' (DR10 bitmasks page)"),
    ("psfsize_g", np.float32, 1, "arcsec", "Weighted average PSF FWHM in the g band"),
    ("psfsize_r", np.float32, 1, "arcsec", "Weighted average PSF FWHM in the r band"),
    ("psfsize_i", np.float32, 1, "arcsec", "Weighted average PSF FWHM in the i band"),
    ("psfsize_z", np.float32, 1, "arcsec", "Weighted average PSF FWHM in the z band"),
    ("psfdepth_g", np.float32, 1, "1/nanomaggy²", "For a 5-sigma point source detection limit in g, 5*sqrt(psfdepth_g) gives flux in nanomaggies and -2.5*log10(5/sqrt(psfdepth_g)) - 9] gives corresponding AB magnitude"),
    ("psfdepth_r", np.float32, 1, "1/nanomaggy²", "For a 5-sigma point source detection limit in r, 5*sqrt(psfdepth_r) gives flux in nanomaggies and -2.5*log10(5/sqrt(psfdepth_r)) - 9] gives corresponding AB magnitude"),
    ("psfdepth_i", np.float32, 1, "1/nanomaggy²", "For a 5-sigma point source detection limit in i, 5*sqrt(psfdepth_i) gives flux in nanomaggies and -2.5*log10(5/sqrt(psfdepth_i)) - 9] gives corresponding AB magnitude"),
    ("psfdepth_z", np.float32, 1, "1/nanomaggy²", "For a 5-sigma point source detection limit in z, 5*sqrt(psfdepth_z) gives flux in nanomaggies and -2.5*log10(5/sqrt(psfdepth_z)) - 9] gives corresponding AB magnitude"),
    ("galdepth_g", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_g' but for a galaxy (0.45\" exp, round) detection sensitivity"),
    ("galdepth_r", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_r' but for a galaxy (0.45\" exp, round) detection sensitivity"),
    ("galdepth_i", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_i' but for a galaxy (0.45\" exp, round) detection sensitivity"),
    ("galdepth_z", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_z' but for a galaxy (0.45\" exp, round) detection sensitivity"),
    ("nea_g", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Noise equivalent area) in g."),
    ("nea_r", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Noise equivalent area) in r."),
    ("nea_i", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Noise equivalent area) in i."),
    ("nea_z", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Noise equivalent area) in z."),
    ("blob_nea_g", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Blob-masked noise equivalent area) in g."),
    ("blob_nea_r", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Blob-masked noise equivalent area) in r."),
    ("blob_nea_i", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Blob-masked noise equivalent area) in i."),
    ("blob_nea_z", np.float32, 1, "arcsec²", "url='../../dr9/nea' (Blob-masked noise equivalent area) in z."),
    ("psfdepth_w1", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_g' (and also on the AB system) but for WISE W1"),
    ("psfdepth_w2", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_g' (and also on the AB system) but for WISE W2"),
    ("psfdepth_w3", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_g' (and also on the AB system) but for WISE W3"),
    ("psfdepth_w4", np.float32, 1, "1/nanomaggy²", "As for 'psfdepth_g' (and also on the AB system) but for WISE W4"),
    ("wise_coadd_id", "S8", 1, "", "unWISE coadd brick name (corresponding to the, <em>e.g.</em>, <cite>legacysurvey-<brick>-image-W1.fits.fz</cite> url='../files/#image-stacks-south-coadd' (coadd file)) for the center of each object"),
    ("wise_x", np.float32, 1, "pix", "X position of coordinates in the brick image stack that corresponds to 'wise_coadd_id' (see the url='../../dr9/updates/#data-model-changes' (DR9 updates page) for transformations between 'wise_x' and 'bx')"),
    ("wise_y", np.float32, 1, "pix", "Y position of coordinates in the brick image stack that corresponds to 'wise_coadd_id' (see the url='../../dr9/updates/#data-model-changes' (DR9 updates page) for transformations between 'wise_y' and 'by')"),
    ("lc_flux_w1", np.float32, 17, "nanomaggy", "flux_w1' in each of up to seventeen unWISE coadd epochs (AB system; defaults to zero for unused entries)"),
    ("lc_flux_w2", np.float32, 17, "nanomaggy", "flux_w2' in each of up to seventeen unWISE coadd epochs (AB; defaults to zero for unused entries)"),
    ("lc_flux_ivar_w1", np.float32, 17, "1/nanomaggy²", "Inverse variance of 'lc_flux_w1' (AB system; defaults to zero for unused entries)"),
    ("lc_flux_ivar_w2", np.float32, 17, "1/nanomaggy²", "Inverse variance of 'lc_flux_w2' (AB; defaults to zero for unused entries)"),
    ("lc_nobs_w1", np.int16, 17, "", "nobs_w1' in each of up to seventeen unWISE coadd epochs"),
    ("lc_nobs_w2", np.int16, 17, "", "nobs_w2' in each of up to seventeen unWISE coadd epochs"),
    ("lc_fracflux_w1", np.float32, 17, "", "fracflux_w1' in each of up to seventeen unWISE coadd epochs (defaults to zero for unused entries)"),
    ("lc_fracflux_w2", np.float32, 17, "", "fracflux_w2' in each of up to seventeen unWISE coadd epochs (defaults to zero for unused entries)"),
    ("lc_rchisq_w1", np.float32, 17, "", "rchisq_w1' in each of up to seventeen unWISE coadd epochs (defaults to zero for unused entries)"),
    ("lc_rchisq_w2", np.float32, 17, "", "rchisq_w2' in each of up to seventeen unWISE coadd epochs (defaults to zero for unused entries)"),
    ("lc_mjd_w1", np.float64, 17, "", "mjd_w1' in each of up to seventeen unWISE coadd epochs (defaults to zero for unused entries)"),
    ("lc_mjd_w2", np.float64, 17, "", "mjd_w2' in each of up to seventeen unWISE coadd epochs (defaults to zero for unused entries)"),
    ("lc_epoch_index_w1", np.int16, 17, "", "Index number of unWISE epoch for W1 (defaults to -1 for unused entries)"),
    ("lc_epoch_index_w2", np.int16, 17, "", "Index number of unWISE epoch for W2 (defaults to -1 for unused entries)"),
    ("sersic", np.float32, 1, "", "Power-law index for the Sersic profile model"),
    ("sersic_ivar", np.float32, 1, "", "Inverse variance of 'sersic'"),
    ("shape_r", np.float32, 1, "arcsec", "Half-light radius of galaxy model for galaxy type 'type' (>0)"),
    ("shape_r_ivar", np.float32, 1, "1/arcsec²", "Inverse variance of 'shape_r'"),
    ("shape_e1", np.float32, 1, "", "Ellipticity component 1 of galaxy model for galaxy type 'type'"),
    ("shape_e1_ivar", np.float32, 1, "", "Inverse variance of 'shape_e1'"),
    ("shape_e2", np.float32, 1, "", "Ellipticity component 2 of galaxy model for galaxy type 'type'"),
    ("shape_e2_ivar", np.float32, 1, "", "Inverse variance of 'shape_e2'"),
)

nJy_per_nmgy = (22.5*u.ABmag).to(u.nJy).value

apertures_band = {}

for values in columns:
    assert len(values) == 5, f"{values[0]} len={len(values)}"
    name, dtype_ref, shape2, unit, desc = values
    if name not in tab_ap.colnames:
        raise ValueError(f"column {name} not found")
    column = tab_ap[name]
    assert column.dtype <= dtype_ref, f"{name} dtype={column.dtype} !<= {dtype_ref}"
    if unit == "nanomaggy":
        unit = "nJy"
        tab_ap[name] *= nJy_per_nmgy
        # The above line makes a new column object for some reason
        column = tab_ap[name]
    shape = column.shape
    if len(shape) == 1:
        assert shape2 == 1
        column.unit = unit
        column.description = desc
    else:
        assert shape2 == shape[1]
        if shape2 == 17:
            # These are WISE columns and not of much interest to us
            pass
        elif name.startswith("apflux_"):
            rest = name[7:]
            if not "_" in rest:
                # This is a flux
                ap_desc = "apertures of radius ["
                if rest[0] == "w":
                    # This is a WISE flux
                    apertures = (3, 5, 7, 9, 11)
                else:
                    # non-WISE flux
                    apertures = (0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.0)
                apertures_band[rest] = apertures
                ap_desc = f"{ap_desc}{', '.join(str(ap) for ap in apertures)}]"
                indexof = desc.find(ap_desc)
                assert indexof > 7, f"{ap_desc} not in {desc=}"
                desc_new = f"{desc[:indexof]}aperture of radius {{radius}}{desc[indexof + len(ap_desc):]}"
            else:
                band = rest.rsplit("_", 1)[1]
                apertures = apertures_band[band]
                desc_new = f"{desc}, radius={radius}"
            for idx in range(shape2):
                radius = apertures[idx]
                name_new = f"{name}_r{str(radius).replace('.', 'p')}"
                tab_ap[name_new] = column.value[:, idx]
                column_new = tab_ap[name_new]
                column_new.description = desc_new
                column_new.unit = unit
        elif name == "dchisq":
            models = ("None", "PSF", "REX", "DEV", "EXP", "SER")
            for idx in range(shape2):
                desc = f"Difference in χ**2 between {models[idx]} and {models[idx + 1]} models"
                name_new = f"dchisq_{models[idx].lower()}_{models[idx + 1].lower()}"
                tab_ap[name_new] = column.value[:, idx]
                tab_ap[name_new].description = desc
        else:
            raise RuntimeError(f"Unexpected {shape=} for column {name}")
        del tab_ap[name]

coords = [
    SpherePoint(ra, dec, degrees) for ra, dec in zip(tab_ap["ra"], tab_ap["dec"])
]
within = np.array([tractInfo.contains(coord) for coord in coords])
if np.sum(within) != len(within):
    tab_ap = tab_ap[within]
    coords = [coord for coord, in_tract in zip(coords, within) if in_tract]
patches = np.array(
    [tractInfo.findPatch(coord).getSequentialIndex() for coord in coords],
    dtype=np.int16,
)
tab_ap["patch"] = patches
tab_ap["patch"].description = f"{skymap} patch index"

tab_arrow = astropy_to_arrow(tab_ap)
row_group_size = compute_row_group_size(tab_arrow.schema)

pq.write_table(tab_arrow, f"{name_tab}.parq", row_group_size=row_group_size)
