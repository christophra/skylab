import ps_model
WeightLLH = ps_model.WeightLLH
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from copy import deepcopy
from itertools import product
from .utils import parabola_window

# integrate this into Skylab (!)
import lightcurve_auto_threshold as lcat

_parab_cache = np.zeros((0, ), dtype=[("S1", np.float), ("a", np.float),
                                      ("b", np.float)])
_par_val = np.nan                                    

class TimePDF(object):
    r""" time pdf, top level class, for given time returns value
    can be either binned pdf or a function (Gaussian)
    """
    def __init__(self,tstart,tend):
        self.tstart=tstart
        self.tend=tend
        self.changed=True # does this have a point (?) is the name appropriate (?)
        
    def tPDFvals(self,ev_times): # what is the point of another name for  the method (?)
        return self.tPDFval_vect(self,ev_times)

class TimePDFBinned(TimePDF):
    r""" the time pdf for the case it is binned, i.e. Fermi lightcurve or a Box.
    takes as imput time_bin_series , which is list of [binstart,value]
    everything before the first bin will be evaluated as zero, the last point should be defining the end of the time PDF 
    and should be of the form [binstart,-1]
    """
    
    def __init__(self,tstart,tend,time_bin_series,threshold=0):
        """Constructor which creates a TimePDFBinned within
            [tstart,tend] : float, float
        by applying
            threshold : float
                the cutoff which the lightcurve has to exceed
        to
            time_bin_series : float, ndarray-like, (N+1) x 2
                [ [lower edge_1, level_1] , ... , [upper edge_N, irrelevant value]]
                
        """
        super(TimePDFBinned, self).__init__(tstart,tend)
        self.TBS=time_bin_series
        self.threshold = threshold
        
    def __str__(self):
        r"""String representation of TimePDFBinned.

        """
        out_str = "{0:s}\n".format(self.__repr__())
        out_str += 67*"~"+"\n"
        out_str += "start time  : {0:d}\n".format(int(self.tstart))
        out_str += "end time  : {0:d}\n".format(int(self.tend))
        out_str += "threshold  : {0:4e}\n".format(self.threshold)
        out_str += 67*"~"+"\n"

        return out_str
    
    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, val):
        self._threshold = val
        self.TBS_to_PDF(val)
    
    def TBS_to_PDF(self,threshold=0):
        r"""Turns list of tuples self.TBS to recarray self.tPDF
        by filling bins and values into the latter as
                    tPDF.t = [tstart,bins]
                    tPDF.v = [0,fluxes,0]
        then selecting tPDF.v above threshold and normalizing."""
        self.tPDF=np.recarray((len(self.TBS)+1,), dtype=[('t', float), ('v', float), ('d', float)])
        self.tPDF.t=np.asarray((self.tstart,)+zip(*self.TBS)[0])
        self.tPDF.v=np.asarray((0.,)+zip(*self.TBS)[1][:-1]+(0.,))
        self.tPDF.d=np.hstack((1.,self.tPDF.t[2:]-self.tPDF.t[1:-1],1.)) # outer values only used when v[0]=v[N]=0
        # self.tPDF.v[self.tPDF.v < self.threshold] = 0. # BlockTimePdf behaviour
        # is same behaviour as BlockTimePdf.
        # BUT used in current Track: BlockTimePdf1
        # which DOES subtract!
        # ergo:
        self.tPDF.v = self.tPDF.v - self.threshold # BlockTimePdf1 behaviour
        self.tPDF.v[self.tPDF.v < 0.] = 0.
        norm=(self.tPDF.v * self.tPDF.d)[:len(self.tPDF)-1].sum()
        #for i in range(len(self.tPDF)-1):
        #    norm+=self.tPDF.v[i]*self.tPDF.d[i]
        self.tPDF.v=self.tPDF.v/norm
        
   
    # try replacing with smoothed tPDF
    @staticmethod
    #@np.vectorize # Vectorizing multiplies overhead (!)
    def tPDFval_vect(self,ev_time):
        r"""Method which evaluates the PDF at ev_time, vectorized."""
        #return self.tPDF.v[max(0,np.searchsorted(self.tPDF.t,ev_time)-1)] # max doesn't handle NumPy arrays (!)
        return self.tPDF.v[np.maximum(0,np.searchsorted(self.tPDF.t,ev_time)-1)]


    @staticmethod
    def tPDFval_smooth(self,ev_time):
        r"""Method which evaluates the PDF at ev_time, vectorized, smoothed with parabola window.
        This however entirely misses the point of getting a smoother behaviour w.r.t. varying thres.
        For that, one would need to include this shape into the LC normalisation instead of replacing each
        block by the same window function that does not change shape."""
        index_i = np.maximum(0,np.searchsorted(self.tPDF.t,ev_time)-1)
        v_i = self.tPDF.v[index_i] # block pdf value
        res_i = (ev_time - self.tPDF.t[index_i])/self.tPDF.d[index_i] # time residual in window
        return parabola_window(res_i)*v_i
        
        
class TimePDFGauss(TimePDF):
    r""" the time pdf for the case of a gaussuan shaped flare """
    
    def __init__(self,tstart,tend):
        super(TimePDFGauss, self).__init__(tstart,tend)        
        self.mean=(tend-tstart)/2. # what is the point of this definition (?)
        self.sigma=self.mean/100. # dito (?)
        
    def setMeanAndSigma(self,mean,sigma):
        self.mean=mean
        self.sigma=sigma
        
    def tPDFvals(self,ev_times):
        gfcn=norm(loc = self.mean, scale = self.sigma)
        return gfcn.pdf(ev_times)
        
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
                thres_seed      : (default: value from lightcurve_auto_threshold)
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
    
    def __init__(self, twodim_bins=ps_model._2dim_bins,timePDF=None, twodim_range=None, thres_division=_thres_division, **kwargs):

        self.timePDF=timePDF
        lc_fluxes = np.array(zip(*self.timePDF.TBS)[1][:-1]) # last entry is not flux
        lc_bins = np.array(zip(*self.timePDF.TBS)[0]) # last entry is last block's upper edge
        lc_fluxes_max = lc_fluxes.max()
        # determine default threshold seed from the lightcurve
        lc_thres_seed  = lcat.auto_threshold(lc_bins, lc_fluxes)
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
        self._w_pdf_dict = dict()
        super(LightcurveLLH, self)._setup(exp)
    
    def __call__(self, exp, mc,livetime):
        r"""In addition to the the splines for energy and declination from WeightLLH,
        create histogram for azimuth and cos(zenith) and time pdfs from threshold grid."""
        
        # set up sinDec spline, and energy spline.
        # calls _setup  to setup everything for energy and time weight calculation.
        super(LightcurveLLH, self).__call__(exp, mc,livetime)

        # calculate time PDFs for grid of threshold values
        # in principle it can be more parameters, so the code is a bit complicated
        par_grid = dict()
        for par, val in self.params.iteritems():
            if par!="thres": # need this since it's hardcoded in weight() - better solution (?)
                continue
            # create grid of threshold values within boundaries
            low, high = val[1]
            # `high` can be flux maximum, for which PDF is undefined
            # => arange does not include `high`
            grid = np.arange(low,
                             high,
                             self._thres_precision)
            par_grid[par] = grid

        pars = par_grid.keys()
        for tup in product(*par_grid.values()):
            # cache the pdf - i.e. set threshold and normalise
            param_dict = dict([(p_i, self._thres_around(t_i)) for p_i, t_i in zip(pars, tup)])
            self._w_pdf_dict[tuple(param_dict.items())] = TimePDFBinned(self.timePDF.tstart,
                                                                  self.timePDF.tend,
                                                                  self.timePDF.TBS,
                                                                  threshold=param_dict["thres"],
                                                                 )
                                           
        
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
        
    
    
    def _get_weights(self, mc, **params):
        r"""Return event weights in MonteCarlo set corresponding to
        only an unnormalized power-law flux E^(-gamma), no other dependency.
        Used for making (splined) histograms of signal weight.
        Parameters:
        -----------
        mc : structured array
                MonteCarlo data
        gamma : float
                Spectral index
        
        Returns:
        -----------
        weights : array-like
                Energy-dependent weights for all MonteCarlo events.
        """
        return mc["ow"] * mc["trueE"]**(-params["gamma"])
    
    def _thres_around(self, value):
        r"""Round a value to the nearest grid point defined in the class.
        This is a different grid than for other parameters, since it's oriented
        on an even division of the params['thres'] bounds.

        Parameters
        -----------
        value : array-like
            Values to round to precision.

        Returns
        --------
        round : array-like
            Rounded values.

        """
        return (np.around(float(value - self.params['thres'][1][0]) / self._thres_precision)\
                * self._thres_precision)\
                 + self.params['thres'][1][0]
    
    def _time_weight(self, ev, **params):
        r"""Evaluate time weights from cached pdf's with `params`.
        Also determine the gradient.
        This computation is separate from the energy weights since they are uncorrelated.
        
        Parameters
        -----------
        ev : structured array
            Events to be evaluated

        **params : float
            Dictionary of parameters, of which M are used for time weights.
            Currently `thres` and `delay`.

        Returns
        --------
        val : array-like (N), N events
            Function value.

        grad : array-like (N, M), N events in M parameter dimensions
            Gradients at function value.
        """
        thres = params["thres"]
        delay = params["delay"]
        
        # evaluate on finite gridpoints in threshold thres
        th1 = self._thres_around(thres)
        dth = self._thres_precision
        # the parabola evaluation doesn't work for outer grid points
        # i.e. [lower bound, upper bound)
        # in which case: define the parabola one step away
        low, high = self.params['thres'][1]
        if th1<=self._thres_around(low):
            self._th1_set_higher_counter += 1
            #print "setting th1 from %4e"%th1,
            th1 = self._thres_around(low + dth)
            #print "to %4e"%th1
        if th1>=self._thres_around(high - dth):
            self._th1_set_lower_counter += 1
            #print "setting th1 from %4e"%th1,
            th1 = self._thres_around(high - 2*dth)
            #print "to %4e"%th1
        # the rest works just the same, only then it's an extrapolation    

        # check whether the grid point of evaluation has changed
        if (np.isfinite(self._th1)
                and th1 == self._th1
                and len(ev) == len(self._time_w_cache)
                and self._previous_delay==delay): # this could fail (!)
            self._used_time_w_cache += 1
            S1 = self._time_w_cache["S1"]
            a = self._time_w_cache["a"]
            b = self._time_w_cache["b"]
            
        else:
            self._filled_time_w_cache += 1
            # evaluate neighbouring gridpoints and parametrize a parabola
            th0 = self._thres_around(th1 - dth)
            th2 = self._thres_around(th1 + dth)

            S0 = self._w_pdf_dict[(("thres", th0), )].tPDFvals(ev["time"] + delay)
            S1 = self._w_pdf_dict[(("thres", th1), )].tPDFvals(ev["time"] + delay)
            S2 = self._w_pdf_dict[(("thres", th2), )].tPDFvals(ev["time"] + delay)

            a = (S0 - 2. * S1 + S2) / (2. * dth**2)
            b = (S2 - S0) / (2. * dth)

            # cache values
            self._th1 = th1

            self._time_w_cache = np.zeros((len(ev),),
                                     dtype=[("S1", np.float), ("a", np.float),
                                            ("b", np.float)])
            self._time_w_cache["S1"] = S1
            self._time_w_cache["a"] = a
            self._time_w_cache["b"] = b
        
        # store previous time delay
        self._previous_delay = delay

        # calculate value at the parabola
        val = a * (thres - th1)**2 + b * (thres - th1) + S1
        grad = 2. * a * (thres - th1) + b
        
        # normalize with flat background time pdf
        w_B =  1./(self.timePDF.tend - self.timePDF.tstart)
        val = val / w_B
        grad = grad / w_B

        return val, np.atleast_2d(grad)
        
    def reset(self):
        r"""Reset all cached values for energy and time weights.

        """
        super(LightcurveLLH, self).reset()
        self._time_w_cache = deepcopy(_parab_cache)

        return    
        
    def weight(self, ev, **params):
        r"""Return weight, gradient from the parent class with an additional
        time-dependent factor on the weight, given by tPDFvals.
        This is the event weight which is evaluated for each event, LLH call.
        Parameters:
        -----------
        ev      : structured array
                Event data. Has to have column "time" for time-dependent weight.
        params  : dict of keyword arguments (optional)
                  Contains threshold for time-dependent weights.
                  Others, like gamma, are passed to WeightLLH.weight.
        Returns:
        -----------
        val     : array
                Weight values corresponding, with factor tPDFvals(ev["time"]).
        grad_w  : array
                Same gradient as from WeightLLH.
        """
        # get time weights and gradients
        t_weight, t_grad = self._time_weight(ev, **params)
        # compute energy weights
        e_weight, e_grad = super(LightcurveLLH, self).weight(ev, **params)
        
        # combine energy and time
        val = e_weight * t_weight # weight is an array => multiply
        grad_w = np.vstack((e_grad, t_grad)) # grad is a 2d array => append
        
        return val, grad_w