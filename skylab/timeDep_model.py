import ps_model
WeightLLH = ps_model.WeightLLH
PowerLawLLH = ps_model.PowerLawLLH
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from copy import deepcopy
from itertools import product
from .utils import parabola_window
import lightcurve_helpers
from .timeDep_pdf import TimePDFBinned

_parab_cache = np.zeros((0, ), dtype=[("S1", np.float), ("a", np.float),
                                      ("b", np.float)])
_par_val = np.nan                                    


        
class LightcurveLLH(WeightLLH):
    r""" Simplest time dep. search likelihood.
        LightcurveLLH is constructed with the arguments
                timePDF         : an instance of TimePDFBinned
                twodim_bins:    : (default: ps_model._2dim_bins)
                twodim_range    : range of (logE, sinDec) passed to WeightLLH.
        Optional keyword arguments are
                gamma_seed      : (default: ps_model._gamma_params["gamma"][0])
                                Seed for spectral index.
                gamma_bounds    : (default: ps_model._gamma_params["gamma"][1])
                                Bounds for spectral index.
                thres_seed      : (default: value from automatic threshold)
                                Seed for lightcurve threshold
                thres_bounds    : (default: (0, maximum flux))
                                Bounds for lightcurve threshold
                delay_seed      : (default: 0.)
                                Seed for lightcurve delay
                delay_bounds    : (default: (-.5, .5) day)
                                Bounds for lightcurve delay
        Extra keyword arguments are passed on to the constructor of WeightLLH.
        """
        
    _time_w_cache = deepcopy(_parab_cache)
    _th1 = _par_val
    _th1_set_lower_counter = 0 # counts how often threshold had to be extrapolated upwards
    _th1_set_higher_counter = 0 # counts how often threshold had to be extrapolated downwards
    _filled_time_w_cache = 0 # counts how often time LLH parabola coeffiecients were computed and stored
    _used_time_w_cache = 0 # counts how often time LLH parabola coefficients were loaded from cache
    _previous_delay = 0. # which delay was set at the previous evaluation of the time weights
    
    _thres_division = 1000 # how many times to divide the lightcuve maximum to get the threshold grid
    
    def __init__(self,
                 twodim_bins=ps_model._2dim_bins,
                 timePDF=None,
                 twodim_range=None,
                 thres_division=_thres_division,
                 **kwargs):

        self.timePDF=timePDF
        lc_fluxes = np.array(zip(*self.timePDF.TBS)[1][:-1]) # last entry is not flux
        lc_bins = np.array(zip(*self.timePDF.TBS)[0]) # last entry is last block's upper edge
        lc_fluxes_max = lc_fluxes.max()
        # determine default threshold seed from the lightcurve
        lc_thres_seed  = lightcurve_helpers.auto_threshold(lc_bins, lc_fluxes)
        # determine default threshold bounds from the max of the lightcurve
        lc_thres_min = 0. # or should it be lc_fluxes.min() (?)
        lc_thres_max = lc_fluxes.max()  
        
        # Defaults for lightcurve delay (vs.the llh) seed and bounds
        lc_delay_seed =  0.
        lc_delay_min  = -.5
        lc_delay_max  =  .5  
        params = dict(gamma=(kwargs.pop("gamma_seed", ps_model._gamma_params["gamma"][0]),
                             deepcopy(kwargs.pop("gamma_bounds", deepcopy(ps_model._gamma_params["gamma"][1])))),
                      thres=(kwargs.pop("thres_seed", lc_thres_seed),
                             deepcopy(kwargs.pop("thres_bounds", (lc_thres_min, lc_thres_max)))),
                      delay=(kwargs.pop("delay_seed", lc_delay_seed),
                             deepcopy(kwargs.pop("delay_bounds", (lc_delay_min, lc_delay_max)))),
                             )
        if params["thres"][1][1] > lc_fluxes_max:
            raise ValueError("Threshold upper bound %4.2e is higher than lightcurve maximum %4.2e"%(params["thres"][1][1], lc_fluxes_max))
        # set threshold grid precision from "division" of smallest interval
        self._thres_division = thres_division # get from kwargs
        self._thres_precision = lc_fluxes_max / self._thres_division
            
        super(LightcurveLLH, self).__init__(params, ["logE", "sinDec"], twodim_bins, range=twodim_range, normed=1, **kwargs)

        r"""this part is similar to the time integrated, but the important difference is that it is in local coodinates
        so that on short time scales the detector special directions are evaluated correctly
        """
        # We don't do this actually because our time scales are always longer than one day

        self.Azimuth_bins_n=15 # is there a good reason these are hardcoded (?)
        self.cosZenith_bins_n=6 # would we ever need to change this (?)
        
    def _setup(self, exp):
        r"""Set up everything for weight calculation.

        """
        #self._w_pdf_dict = dict()
        super(LightcurveLLH, self)._setup(exp)
    
    def __call__(self, exp, mc,livetime):
        r"""In addition to the the splines for energy and declination from WeightLLH,
        create histogram for azimuth and cos(zenith) and time pdfs from threshold grid."""
        
        # set up sinDec spline, and energy spline.
        # calls _setup  to setup everything for energy and time weight calculation.
        super(LightcurveLLH, self).__call__(exp, mc,livetime)
                                           
        
        # histogram exp data in Azimuth, cosZenith.
        self.hist, binsa, binsd = np.histogram2d(exp["Azimuth"],exp["sinDec"],
                                                 bins=[self.Azimuth_bins_n,self.cosZenith_bins_n],
                                                 range=None, normed=False )
                                                 
        # normalize within sin(declination) bins scaled with Azimuth_bins_n
        # so that  self.hist.mean()==1
        # That way, neglectic azimuth & zenith in signal(...) at least
        # does not introduce big numerical errors.
        for i_dec in range(self.cosZenith_bins_n):
            sumDband=sum(self.hist[:,i_dec])
            for i_azi in range(self.Azimuth_bins_n):
                self.hist[i_azi][i_dec]=self.hist[i_azi][i_dec]/sumDband

        # overwrite range and bins to actual bin edges
        # (chosen with exp data, and self.{Azimuth,cosZenith}_bins_n from __init__)
        # In background_vect this is used to determine bin numbers.
        self.Azimuth_bins = binsa
        self.Azimuth_range = (binsa[0], binsa[-1])
        self.cosZenith_bins= binsd
        self.cosZenith_range = (binsd[0], binsd[-1])

        # check that no bins in hist are 0, so that we can take log(hist) later
        if np.any(self.hist <= 0.):
            estr = ("Local coord. hist bins empty, this must not happen! "
                    +"Empty bins: {0}".format(bmids[self.hist <= 0.]))
            raise ValueError(estr)
        
        # set up background spline that interpolates np.log(hist) between bin centers
        # Currently not used in background(...)
        self._bckg_spline2D = RectBivariateSpline( (binsa[1:] + binsa[:-1]) / 2.,(binsd[1:] + binsd[:-1]) / 2.,  np.log(self.hist))

        
    
    # in case we need local coordinates - not used otherwise
    # find a way to NOT use vectorize (!)        
    @staticmethod
    @np.vectorize
    def background_vect(self, az,zen,sinDec=None):
        r"""Spatial background distribution. 
        No local coordinates for time dep since we are on >1day timescales (!)
        
        For IceCube this is only declination dependent.
        In a more general scenario it is dependent on zenith and zimuth,
        e.g. in ANTARES, KM3NET, or using time dependent information.

        Parameters
        -----------
        az : array-like
            array of event azimuth data
        zen : array-like
            array of event cos(zenith) data
        sinDec : array-like (optional)
                if defined, multiply np.exp(self.bckg_spline(sinDec)) to return value (default: None)

        Returns
        --------
        P : array-like
            background pdf values for az,zen and sinDec (if supplied).
            Not interpolated in az,zen (uses self.hist), splined in sinDec (uses self.bckg_spline)

        """
        # turn (az, zen) into bin numbers using {cosZenith,Azimuth}_{range,bins_n}
        azbin =int(\
                   (az -  self.Azimuth_range[0])\
                   / (self.Azimuth_range[1]-self.Azimuth_range[0])\
                   * self.Azimuth_bins_n\
                  )
        cosbin=int(\
                   (zen - self.cosZenith_range[0])\
                   / (self.cosZenith_range[1] - self.cosZenith_range[0])\
                   * self.cosZenith_bins_n\
                  )
                  
        if azbin==self.Azimuth_bins_n: # include last upper bin edge
            azbin=self.Azimuth_bins_n-1 # else there'll be an indexing error
        if cosbin==self.cosZenith_bins_n:
            cosbin=self.cosZenith_bins_n-1
        if sinDec:
            # beware: self.bckg_spline (which is just a property equal to self._bckg_spline)
            # is actually a spline of the logarithm of the "density=True" histogram
            # hence we need to take the exponential!
            return 1. / 2. / np.pi * np.exp(self.bckg_spline(sinDec)) * self.hist[azbin][cosbin]
        else:
            return self.hist[azbin][cosbin]            

        
    def background(self, ev):
        r"""Return spatial background pdf, evaluated on structured array `ev`."""
        #flatTimePDF=1./(self.timePDF.tend - self.timePDF.tstart) # included in _time_weight (!)
        #return self.background_vect(ev["Azimuth"spatial ],ev["sinDec"],ev["sinDec"]) # don't use local coords (!)
        return 1. / 2. / np.pi * np.exp(self.bckg_spline(ev['sinDec']))
    
    def signal(self, src_ra, src_dec, src_lc, ev, **params):
        r"""Space-time likelihood for events given source position and lightcurve

        Signal is assumed to cluster around source position.
        The distribution is assumed to be well approximated by a gaussian
        locally.
        The times of the signal are assumed to be proportional to the lightcurve.

        Parameters
        -----------
        src_ra : float
            Source right ascension (radians).
        src_dec : float
            Source declination (radians).
        src_lc : TimePDF or derived classes
            Time PDF assumed for the source.
        ev : structured array
            Event array, import information: sinDec, ra, sigma, time.
        params : dict
            Further parameters to the signal likelihood. ATTN: Currently useless,
            since signal(...) is not evaluated again during LLH minimization.

        Returns
        --------
        P : array-like
            Spatial signal probability for each event

        """
        #convert src_ra, dec to numpy arrays if not already done
        src_ra = np.atleast_1d(src_ra)[:,np.newaxis]
        src_dec = np.atleast_1d(src_dec)[:,np.newaxis]
        
        # same for the lightcurve
        if not isinstance(src_lc, list):
            src_lc = [src_lc]
            
        assert len(src_dec)==len(src_lc)
        
        S_spatial  = super(LightcurveLLH, self).signal(src_ra, src_dec, src_lc, ev,**params)
        
        S_temporal = np.zeros_like(S_spatial) # has same shape: sources x events
        for i_source in range(len(src_lc)):
            S_temporal[i_source] = src_lc[i_source].tPDFvals(ev["time"],**params)
        
        return S_spatial*S_temporal
        
        
    def _effA(self, mc, livetime, **pars):
        r"""Construct two dimensional spline of effective area versus
        declination and spectral index for Monte Carlo. This is stored in
        self._spl_effA and later used for calling effA.
        """
    
        gamma_vals = pars["gamma"]

        x = np.sin(mc["trueDec"])
        hist = np.vstack([np.histogram(x,
                                       weights=self._get_weights(mc, gamma=gm)
                                                * livetime * 86400.,
                                       bins=self.sinDec_bins)[0]
                          for gm in gamma_vals]).T

        self._spl_effA = RectBivariateSpline(
                (self.sinDec_bins[1:] + self.sinDec_bins[:-1]), gamma_vals,
                np.log(hist), kx=2, ky=2, s=0)

        return
    
    def effA(self, dec, **params):
        r"""Evaluate effective area at declination and spectral index.

        Parameters
        -----------
        dec : float
            Declination.

        gamma : float
            Spectral index.

        Returns
        --------
        effA : float
            Effective area at given point(s).
        grad_effA : float
            Gradient at given point(s).

        """
        if (np.sin(dec) < self.sinDec_bins[0]
                or np.sin(dec) > self.sinDec_bins[-1]):
            return 0., None

        gamma = params["gamma"]

        val = np.exp(self._spl_effA(np.sin(dec), gamma, grid=False, dy=0.))
        grad = val * self._spl_effA(np.sin(dec), gamma, grid=False, dy=1.)

        return val, dict(gamma=grad)
        
    
        
    def reset(self):
        r"""Reset all cached values for energy and time weights.

        """
        super(LightcurveLLH, self).reset()
        self._time_w_cache = deepcopy(_parab_cache)

        return
