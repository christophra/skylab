from .ps_model import logger, WeightLLH, PowerLawLLH
from .ps_model import _precision, _parab_cache, _par_val, _2dim_bins
#import ps_model
#WeightLLH = ps_model.WeightLLH
#PowerLawLLH = ps_model.PowerLawLLH
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from copy import deepcopy
from itertools import product
from .utils import parabola_window
import lightcurve_helpers
from .timeDep_pdf import TimePDFBinned

#_parab_cache = np.zeros((0, ), dtype=[("S1", np.float), ("a", np.float),
#                                      ("b", np.float)])
#_par_val = np.nan                                    


        
class LightcurveLLH(PowerLawLLH):
    r""" Simplest time dep. search likelihood.
        Optional keyword arguments are
                seed      : (default: ps_model._gamma_params["gamma"][0])
                                Seed for spectral index.
                bounds    : (default: ps_model._gamma_params["gamma"][1])
                                Bounds for spectral index.
        Extra keyword arguments are passed on to the constructor of WeightLLH.
        """

    def __init__(self, *args, **kwargs):
        
        twodim_bins = kwargs.pop("twodim_bins",[_2dim_bins,_2dim_bins])
        twodim_range = kwargs.pop("twodim_range", None)
        super(LightcurveLLH, self).__init__(["logE", "sinDec"], bins = twodim_bins, range=twodim_range, normed=1, **kwargs)

        r"""this part is similar to the time integrated, but the important difference is that it is in local coodinates
        so that on short time scales the detector special directions are evaluated correctly
        """
        # We don't do this actually because our time scales are always longer than one day

        self.Azimuth_bins_n=15 # is there a good reason these are hardcoded (?)
        self.cosZenith_bins_n=6 # would we ever need to change this (?)
        
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
    def _background_vect(self, az,zen,sinDec=None):
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
        #return self._background_vect(ev["Azimuth"spatial ],ev["sinDec"],ev["sinDec"]) # don't use local coords (!)
        return super(LightcurveLLH, self).background(ev)
    
    def _signal_time(self, src_lc, ev, **params):
        r"""Signal time likelihood for a given list of lightcurves and event data.
        
        The signal is assumed to be distributed according to these time pdfs.
        
        Parameters
        -----------
        src_lc : list of TimePDF (or derived classes)
            The time PDFs of the sources.
        ev : structured array
            Event array. Important column: time.
            
        Optional
        -----------
        params : dict (keywords)
            Additional parameters passed to the time PDFs. ATTN: Currently useless,
            since signal(...) is not evaluted during the minimisation, but before.
        """
        
        S_temporal = np.zeros(shape=(len(src_lc), ev.size)) # has shape: sources x events
        for i_source in range(len(src_lc)):
            S_temporal[i_source] = src_lc[i_source].tPDFvals(ev["time"],**params)
        return S_temporal

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
            Event array, important information: sinDec, ra, sigma, time.
        params : dict
            Further parameters to the signal likelihood. ATTN: Currently useless,
            since signal(...) is not evaluated again during LLH minimization.

        Returns
        --------
        P : array-like
            Spatial signal probability for each event and source.

        """
        
        #convert src_ra, dec to numpy arrays if not already done
        src_ra = np.atleast_1d(src_ra)
        src_dec = np.atleast_1d(src_dec)
        
        # same for the lightcurve
        if not isinstance(src_lc, list):
            src_lc = [src_lc for d in src_ra]
        
        S_spatial  = super(LightcurveLLH, self).signal(src_ra, src_dec, ev, **params)
        S_temporal = self._signal_time(src_lc, ev, **params)
        
        return S_spatial*S_temporal
        
        def fast_signal(self, src_ra, src_dec, src_lc, ev, ind, **params):
            r"""Space-time LLH corresponding to ClassicLLH.fast_signal.
            """
            
            if not isinstance(src_lc, list):
                src_lc = [src_lc for d in src_ra]
                
            S_spatial = super(LightcurveLLH, self).fast_signal(src_ra, src_dec, ev)
            S_temporal = self._signal_time(src_lc, np.take(ev,ind), **params)
            
            return S_spatial * S_temporal
            
    
        
    def reset(self):
        r"""Reset all cached values for energy and time weights.

        """
        super(LightcurveLLH, self).reset()
        #self._time_w_cache = deepcopy(_parab_cache)

        return
