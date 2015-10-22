import ps_model
WeightLLH = ps_model.WeightLLH
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from copy import deepcopy

class TimePDF(object):
    r""" time pdf, top level class, for given time returns value
    can be either binned pdf or a function (Gaussian)
    """
    def __init__(self,tstart,tend):
        self.tstart=tstart
        self.tend=tend
        self.changed=True
        
    def tPDFvals(self,ev_times):
        return self.tPDFval_vect(self,ev_times)

class TimePDFBinned(TimePDF):
    r""" the time pdf for the case it is binned, i.e. Fermi lightcurve or a Box.
    takes as imput time_bin_series , which is list of [binstart,value]
    everything before the first bin will be evaluated as zero, the last point should be defining the end of the time PDF 
    and should be of the form [binstart,-1]
    """
    
    def __init__(self,tstart,tend,time_bin_series,threshold=0):
        super(TimePDFBinned, self).__init__(tstart,tend)
        self.TBS=time_bin_series
        self.TBS_to_PDF(threshold)
    
    def TBS_to_PDF(self,threshold=0):
        self.threshold=threshold
        self.tPDF=np.recarray((len(self.TBS)+1,), dtype=[('t', float), ('v', float)])
        self.tPDF.t=np.asarray((self.tstart,)+zip(*self.TBS)[0])
        self.tPDF.v=np.asarray((0.,)+zip(*self.TBS)[1][:-1]+(0.,))
        self.tPDF.v[self.tPDF.v < self.threshold] = 0.
        norm=0.
        for i in range(len(self.tPDF)-1):
            norm+=self.tPDF.v[i]*(self.tPDF.t[i+1]-self.tPDF.t[i])
        self.tPDF.v=self.tPDF.v/norm
        
    @staticmethod
    @np.vectorize
    def tPDFval_vect(self,ev_time):
        return self.tPDF.v[max(0,np.searchsorted(self.tPDF.t,ev_time)-1)]
        
class TimePDFGauss(TimePDF):
    r""" the time pdf for the case of a gaussuan shaped flare """
    
    def __init__(self,tstart,tend):
        super(TimePDFGauss, self).__init__(tstart,tend)        
        self.mean=(tend-tstart)/2.
        self.sigma=self.mean/100.
        
    def setMeanAndSigma(self,mean,sigma):
        self.mean=mean
        self.sigma=sigma
        
    def tPDFvals(self,ev_times):
        gfcn=norm(loc = self.mean, scale = self.sigma)
        return gfcn.pdf(ev_times)
        
class TimeBoxLLH(WeightLLH):
    r""" Simplest time dep. search, a time PDF shaped as a box"""
    
    def __init__(self, twodim_bins=ps_model._2dim_bins,timePDF=None, twodim_range=None, **kwargs):
        
        self.timePDF=timePDF
        params = dict(gamma=(kwargs.pop("seed", ps_model._gamma_params["gamma"][0]),
                             deepcopy(kwargs.pop("bounds", deepcopy(ps_model._gamma_params["gamma"][1])))))

        super(TimeBoxLLH, self).__init__(params, ["logE", "sinDec"],
                                        twodim_bins, range=twodim_range,
                                        normed=1, **kwargs)

        r"""this part is similar to the time integrated, but the important difference is that it is in local coodinates
        so that on short time scales the detector special directions are evaluated correctly
        """
        self.Azimuth_bins=90
        self.cosZenith_bins=12
        
    def __call__(self, exp, mc):
        super(TimeBoxLLH, self).__call__(exp, mc)
        hist, binsa, binsz=np.histogram2d(exp["Azimuth"],exp["cozZenith"], bins=[self.Azimuth_bins,self.cosZenith_bins], range=None, normed=True )

        # overwrite range and bins to actual bin edges
        self.Azimuth_bins = binsa
        self.Azimuth_range = (binsa[0], binsa[-1])
        self.cosZenith_bins= binsz
        self.cosZenith_range = (binsz[0], binsz[-1])

        if np.any(hist <= 0.):
            estr = ("Local coord. hist bins empty, this must not happen! "
                    +"Empty bins: {0}".format(bmids[hist <= 0.]))
            raise ValueError(estr)

        self._bckg_spline = RectBivariateSpline( (binsa[1:] + binsa[:-1]) / 2.,(binsz[1:] + binsz[:-1]) / 2.,  np.log(hist))
        
    
    def background(self, ev):
        flatTimePDF=1./(self.timePDF.tend*self.timePDF.tstart)
        return self.background_vect(self, ev["Azimuth"],ev["cozZenith"])*flatTimePDF
        
    @staticmethod
    @np.vectorize
    def background_vect(self, az,zen):
        r"""Spatial background distribution. local coordinates for time dep.

        For IceCube is only declination dependent, in a more general scenario,
        it is dependent on zenith and
        azimuth, e.g. in ANTARES, KM3NET, or using time dependent information.

        Parameters
        -----------
        ev : structured array
            Event array, importand information *sinDec* for this calculation

        Returns
        --------
        P : array-like

        """
        return np.exp(self.bckg_spline(az,zen))

    def _get_weights(self, **params):
        r"""Calculate weights using the given parameters.

        Parameters
        -----------
        params : dict
            Dictionary containing the parameter values for the weighting.

        Returns
        --------
        weights : array-like
            Weights for each event

        """

        return self._mc["ow"] * self._mc["trueE"]**(-params["gamma"])

    def weight(self, ev, **params):
        print params
        val, grad_w=super(TimeBoxLLH, self).weight(ev, **params)
        if self.timePDF.changed:
            self.time_w=self.timePDF.tPDFvals(ev["mjd"])
        
        val=val*self.time_w
        
        return val, grad_w
    
    
    